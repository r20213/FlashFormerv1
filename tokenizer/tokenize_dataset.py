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
    .pip_install("transformers", "tokenizers", "numpy", "datasets", "fsspec", "tqdm")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("config")
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
    Independent parallel worker block. Uses a lightweight stream generator filter 
    to guarantee zero row collisions based purely on explicit worker_id index mapping.
    """
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    
    print(f"👷 [Worker {worker_id}/{num_workers}] Initializing tokenizer layer...")
    tokenizer = AutoTokenizer.from_pretrained(C.HUB_REPO_ID, token=os.environ["HF_TOKEN"])
    
    raw_stream = load_dataset(
        C.NEMOTRON_DATASET,
        name=C.NEMOTRON_SUBSET,
        split=C.NEMOTRON_SPLIT,
        streaming=True,
        token=os.environ["HF_TOKEN"]
    )
    
    # Mathematical isolation: Ensures workers process mutually exclusive lines
    def worker_isolated_stream(iterable):
        for idx, element in enumerate(iterable):
            if idx % num_workers == worker_id:
                yield element

    dataset = worker_isolated_stream(raw_stream)
    shard_file_path = f"/data/tokens_shard_{worker_id}.bin"
    
    tokens_saved_count = 0
    lvl4_processed = 0
    lvl5_processed = 0
    buffer_ids = []
    FLUSH_THRESHOLD = 500_000 
    
    progress_bar = tqdm(
        dataset,
        desc=f"👷 Shard {worker_id}",
        mininterval=15.0, # Keeps output scannable without flooding terminal frames
        unit=" rows"
    )
    
    with open(shard_file_path, "wb") as f_out:
        for row in progress_bar:
            metadata = row.get("metadata", {})
            score = metadata.get("finemath_int_scores", 0)
            
            if score == 4:
                lvl4_processed += 1
            elif score == 5:
                lvl5_processed += 1
            else:
                continue
                
            raw_text = row.get(C.NEMOTRON_TEXT_COL, "")
            if not raw_text:
                continue
                
            enc = tokenizer(raw_text, add_special_tokens=False)
            buffer_ids.extend(enc["input_ids"])
            
            # Flush tokens to disk at regular thresholds to preserve memory overhead
            if len(buffer_ids) >= FLUSH_THRESHOLD:
                np_array = np.array(buffer_ids, dtype=np.uint16)
                f_out.write(np_array.tobytes())
                tokens_saved_count += len(np_array)
                buffer_ids.clear()
                f_out.flush()
                
                # Streaming Output tracking: Prints directly through Modal's async engine logs
                print(f"⭐ [Progress Report] Worker {worker_id} pushed {tokens_saved_count:,} total tokens to disk storage layer.", flush=True)
                progress_bar.set_postfix({"tokens": f"{tokens_saved_count:,}"})
                
        # Final residual cache sweep
        if buffer_ids:
            np_array = np.array(buffer_ids, dtype=np.uint16)
            f_out.write(np_array.tobytes())
            tokens_saved_count += len(np_array)
            buffer_ids.clear()
            
    print(f"💾 [Worker {worker_id}] Processing finalized cleanly. Saved: {tokens_saved_count:,} tokens.")
    
    # Force instant flush from container memory straight to central volume
    data_volume.commit()
    
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
    
    print("✨ Launching 100 parallel workers across cluster matrix...\n")
    
    # starmap returns a generator that consumes worker stdout logs on the fly
    for partial_count, l4, l5 in tokenize_dataset_shard.starmap(worker_inputs, order_outputs=False):
        total_tokens_accumulated += partial_count
        aggregate_lvl4 += l4
        aggregate_lvl5 += l5
        
        print(f"📈 Shard Closed. Global Counter Matrix: {total_tokens_accumulated:,} / {C.PRETRAIN_TARGET_TOKENS:,} tokens.")
        
        if total_tokens_accumulated >= C.PRETRAIN_TARGET_TOKENS:
            print(f"\n🎯 Target Matrix Limit of {C.PRETRAIN_TARGET_TOKENS:,} successfully reached!")
            break
            
    print("\n🏁 Distributed Processing Phase Finished.")
    print(f"📊 Aggregate Processed Corpus Capacity: {total_tokens_accumulated:,} tokens.")
    print(f"📈 Total Level 4 Documents Processed: {aggregate_lvl4:,}")
    print(f"📈 Total Level 5 Documents Processed: {aggregate_lvl5:,}")
    print(f"⏱️ Matrix Run Duration: {timedelta(seconds=int(time.time() - t0))}")