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

# FIX: Bind a lifecycle-managed cloud Queue directly to the app instance layout
app.global_metrics_queue = modal.Queue.ephemeral()

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
    Independent parallel worker block. Pipes performance metrics back to the
    orchestrator in real time using the app metric queue layout.
    """
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    
    # Extract the bound queue resource straight from the active execution context
    q = app.global_metrics_queue
    
    print(f"👷 [Worker {worker_id}/{num_workers}] Initializing tokenizer layer...")
    tokenizer = AutoTokenizer.from_pretrained(C.HUB_REPO_ID, token=os.environ["HF_TOKEN"])
    
    raw_stream = load_dataset(
        C.NEMOTRON_DATASET,
        name=C.NEMOTRON_SUBSET,
        split=C.NEMOTRON_SPLIT,
        streaming=True,
        token=os.environ["HF_TOKEN"]
    )
    
    # Strict index extraction isolation
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
        mininterval=15.0,
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
            
            if len(buffer_ids) >= FLUSH_THRESHOLD:
                np_array = np.array(buffer_ids, dtype=np.uint16)
                f_out.write(np_array.tobytes())
                
                # Metrics Package
                delta_tokens = len(np_array)
                tokens_saved_count += delta_tokens
                buffer_ids.clear()
                f_out.flush()
                
                # Send performance pack straight into the app loop structure
                q.put((delta_tokens, score == 4, score == 5, False))
                progress_bar.set_postfix({"tokens": f"{tokens_saved_count:,}"})
                
        # Final residual cache sweep
        if buffer_ids:
            np_array = np.array(buffer_ids, dtype=np.uint16)
            f_out.write(np_array.tobytes())
            delta_tokens = len(np_array)
            tokens_saved_count += delta_tokens
            buffer_ids.clear()
            q.put((delta_tokens, False, False, False))
            
    print(f"💾 [Worker {worker_id}] Tokenization completed. Saved: {tokens_saved_count:,} tokens.")
    data_volume.commit()
    
    # Notify main thread that this worker is completely done
    q.put((0, 0, 0, True))
    return tokens_saved_count

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
    
    q = app.global_metrics_queue
    worker_inputs = [(i, NUM_CONCURRENT_WORKERS) for i in range(NUM_CONCURRENT_WORKERS)]
    
    # Fire off all workers asynchronously in the background background
    tokenize_dataset_shard.starmap(worker_inputs, order_outputs=False)
    
    total_tokens_accumulated = 0
    aggregate_lvl4 = 0
    aggregate_lvl5 = 0
    finished_workers = 0
    
    print("✨ Workers deployed to cluster. Listening for stream data packets...\n")
    
    # Continuous listening thread layout
    while finished_workers < NUM_CONCURRENT_WORKERS:
        try:
            # Pull metrics directly as they appear from any cloud worker
            delta_tokens, is_l4, is_l5, is_done = q.get(timeout=5)
            
            if is_done:
                finished_workers += 1
                continue
                
            total_tokens_accumulated += delta_tokens
            if is_l4: aggregate_lvl4 += 1
            if is_l5: aggregate_lvl5 += 1
            
            # Print feedback instantly the second ANY worker flushes data!
            print(f"📈 Global Counter: {total_tokens_accumulated:,} / {C.PRETRAIN_TARGET_TOKENS:,} tokens | Active Workers: {NUM_CONCURRENT_WORKERS - finished_workers}/100")
            
            if total_tokens_accumulated >= C.PRETRAIN_TARGET_TOKENS:
                print(f"\n🎯 Target Matrix Limit of {C.PRETRAIN_TARGET_TOKENS:,} successfully reached!")
                break
                
        except Exception:
            # Keep loop alive if queue is momentarily empty
            continue
            
    print("\n🏁 Distributed Processing Phase Finished.")
    print(f"📊 Aggregate Processed Corpus Capacity: {total_tokens_accumulated:,} tokens.")
    print(f"⏱️ Matrix Run Duration: {timedelta(seconds=int(time.time() - t0))}")