import os
import sys
import time
from datetime import timedelta
import config as C

# 1. Immediate validation of Hugging Face orchestration credentials
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ Critical Error: HF_TOKEN environment variable not set. Exiting immediately.")
    sys.exit(1)

import modal

# 2. Programmatically initialize and secure the Modal Secrets Layer
try:
    tokenizer_secret = modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})
except Exception as e:
    print(f"❌ Failed to initialize Modal Secret structural mapping: {e}")
    sys.exit(1)

# Define hardware cluster parameters matching your exact operational spec
image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "tokenizers", "numpy", "datasets", "fsspec")
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("mathformer-distributed-pretrain-pack", image=image)
data_volume = modal.Volume.from_name(C.TOKENIZED_DATA_VOLUME, create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────────
# Worker Function
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    cpu=2.0,
    memory=4096,
    secrets=[tokenizer_secret],
    volumes={"/data": data_volume},
    timeout=7200
)
def tokenize_dataset_shard(worker_id: int, num_workers: int):
    """
    Independent parallel worker block. Streams data deterministically using a
    modulo identity matrix to completely eliminate cross-worker row collisions.
    Balances lvl 4 and lvl 5 data dynamically on the fly to a 50/50 split.
    """
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random
    
    print(f"👷 [Worker {worker_id}/{num_workers}] Initializing tokenizer layer...")
    # Pulling directly from your verified repository deployment asset
    tokenizer = AutoTokenizer.from_pretrained(C.HUB_REPO_ID, token=os.environ["HF_TOKEN"])
    
    print(f"🎯 [Worker {worker_id}/{num_workers}] Connecting to streaming dataset slice...")
    dataset = load_dataset(
        C.NEMOTRON_DATASET,
        name=C.NEMOTRON_SUBSET,
        split=C.NEMOTRON_SPLIT,
        streaming=True,
        token=os.environ["HF_TOKEN"]
    )
    
    # Isolate individual worker sequence spaces to ensure uniform, deterministic skipping
    random.seed(42 + worker_id)
    
    # Keeping ~14% of Level 4 items downsamples their heavy presence inside the 
    # 4plus subset to match the scarcer Level 5 count perfectly.
    LVL4_KEEP_PROB = 0.14
    
    # Direct binary append stream - optimal memory throughput layout
    shard_file_path = f"/data/tokens_shard_{worker_id}.bin"
    
    tokens_saved_count = 0
    lvl4_processed = 0
    lvl5_processed = 0
    buffer_ids = []
    FLUSH_THRESHOLD = 500_000 # Memory-flush gate to balance disk I/O
    
    with open(shard_file_path, "wb") as f_out:
        # Enumerate stream to isolate deterministic unique row indices
        for idx, row in enumerate(dataset):
            # 6. Strict Worker Isolation Rule (No double processing)
            if idx % num_workers != worker_id:
                continue
                
            # Filter rows based on metadata schema criteria
            metadata = row.get("metadata", {})
            score = metadata.get("finemath_int_scores", 0)
            
            # Balance Level 4 and Level 5 rows on the fly
            if score == 4:
                if random.random() > LVL4_KEEP_PROB:
                    continue
                lvl4_processed += 1
            elif score == 5:
                lvl5_processed += 1
            else:
                # Fallback guard against unexpected metadata entries
                continue
                
            raw_text = row.get(C.NEMOTRON_TEXT_COL, "")
            if not raw_text:
                continue
                
            # Run lightning-fast C-level layout execution pass
            enc = tokenizer(raw_text, add_special_tokens=False)
            buffer_ids.extend(enc["input_ids"])
            
            # Efficient block-streaming I/O
            if len(buffer_ids) >= FLUSH_THRESHOLD:
                np_array = np.array(buffer_ids, dtype=np.uint16)
                f_out.write(np_array.tobytes())
                tokens_saved_count += len(np_array)
                buffer_ids.clear()
                
        # Final residual cache sweep
        if buffer_ids:
            np_array = np.array(buffer_ids, dtype=np.uint16)
            f_out.write(np_array.tobytes())
            tokens_saved_count += len(np_array)
            buffer_ids.clear()
            
    print(f"💾 [Worker {worker_id}] Tokenization phase terminated. Saved: {tokens_saved_count:,} tokens (Lvl4 rows: {lvl4_processed:,}, Lvl5 rows: {lvl5_processed:,}).")
    return tokens_saved_count, lvl4_processed, lvl5_processed

# ─────────────────────────────────────────────────────────────────────────────
# Local Orchestration Coordinator
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    t0 = time.time()
    NUM_CONCURRENT_WORKERS = 100
    
    print("🚀 Initiating parallel map orchestrator...")
    print(f"📦 Mount Volume: {C.TOKENIZED_DATA_VOLUME}")
    print(f"📊 Extraction Matrix Target: {C.PRETRAIN_TARGET_TOKENS:,} total tokens.")
    
    worker_inputs = [(i, NUM_CONCURRENT_WORKERS) for i in range(NUM_CONCURRENT_WORKERS)]
    
    total_tokens_accumulated = 0
    aggregate_lvl4 = 0
    aggregate_lvl5 = 0
    
    # Gather execution tracking from all workers asynchronously
    for partial_count, l4, l5 in tokenize_dataset_shard.starmap(worker_inputs, order_preserved=False):
        total_tokens_accumulated += partial_count
        aggregate_lvl4 += l4
        aggregate_lvl5 += l5
        
        # Immediate tracking check against configuration parameters
        if total_tokens_accumulated >= C.PRETRAIN_TARGET_TOKENS:
            print(f"\n🎯 Target Matrix Limit of {C.PRETRAIN_TARGET_TOKENS:,} reached.")
            break
            
    print("\n🏁 Distributed Processing Phase Finished.")
    print(f"📊 Aggregate Processed Corpus Capacity: {total_tokens_accumulated:,} tokens.")
    print(f"📈 Total Balanced Level 4 Documents: {aggregate_lvl4:,}")
    print(f"📈 Total Balanced Level 5 Documents: {aggregate_lvl5:,}")
    print(f"⏱️ Matrix Run Duration: {timedelta(seconds=int(time.time() - t0))}")