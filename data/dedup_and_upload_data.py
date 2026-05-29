import os
import sys
import argparse
import numpy as np
import collections
import hashlib
from typing import Optional

# Core imports
from datasets import load_dataset, interleave_datasets, Dataset
import sentencepiece as spm
from detoxify import Detoxify
import fasttext

# Performance Tuning for 4 vCPU environments (Prevents PyTorch from oversaturating CPU threads)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Force path resolution for local configurations
sys.path.append(os.getcwd())
try:
    from config import DataConfig  # Ingests your exact architecture layouts
except ImportError:
    raise ImportError("Could not import DataConfig from config.py. Ensure config.py exists in the current directory.")

# --- LOCAL DEPLOYMENT TUNING PARAMETERS ---
TARGET_TOKENS = 20_000_000_000     # 20 Billion tokens total
SHARD_SIZE_TOKENS = 100_000_000   # 100M tokens per shard (~200 files total)
DEDUP_WINDOW_SIZE = 100_000       # Tracks last 100k document hashes seamlessly

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-Optimized Dataset Tokenization and Pack Pipeline")
    parser.add_argument(
        "--hf_export_repo", 
        type=str, 
        required=True, 
        help="Mandatory Hugging Face repository destination (e.g., 'username/prefiltered-packed-235m')"
    )
    return parser.parse_args()

def _extract_text_by_config(row: dict, src_config: dict) -> Optional[str]:
    """Extracts text strings adhering exactly to DataConfig structural keys."""
    path = src_config["path"]
    col = src_config.get("text_column")

    if path == "openbmb/UltraInteract_sft":
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        if not instruction or not response:
            return None
        return f"{instruction}\n\n{response}"

    if col and col in row:
        val = row[col]
        if val and isinstance(val, str):
            return val.strip()
    return None

def build_and_upload_dataset(hf_export_repo: str):
    print("Initializing DataConfig Specifications...")
    dcfg = DataConfig()

    # Explicitly override the legacy 'c++' directory key to prevent Hub routing crashes
    for src in dcfg.sources:
        if src.get("path") == "bigcode/starcoderdata" and src.get("name") == "c++":
            src["name"] = "cpp"

    print("Loading Multi-GPU & CPU Filter Handlers...")
    
    # 2x T4 GPU Optimization: Dual-Model Pipeline Round-Robin Setup
    # Instead of multi-processing (which copies the 27GB RAM space and crashes your system),
    # we instantiate two model references targeted at different cuda devices on the main thread.
    print(" -> Allocating Detoxify Models to cuda:0 and cuda:1...")
    detox_gpu0 = Detoxify("original", device="cuda:0")
    detox_gpu1 = Detoxify("original", device="cuda:1")
    detox_models = [detox_gpu0, detox_gpu1]
    gpu_selector = 0  # Round-robin toggle pointer

    ft_model = fasttext.load_model("tokenizer/fasttext_model.bin")
    sp = spm.SentencePieceProcessor(model_file=dcfg.tokenizer_path)

    sources = dcfg.sources
    hf_token = os.environ.get("HF_TOKEN", None)
    hf_datasets = []
    
    print("Binding streaming generators to HuggingFace Hub repositories...")
    for src in sources:
        kwargs = {
            "streaming": True,
            "split": src["split"],
            "trust_remote_code": True,
            "token": hf_token,
        }
        if src["path"] == "bigcode/starcoderdata":
            kwargs["data_dir"] = src["name"]
        elif src.get("name") is not None:
            kwargs["name"] = src["name"]
            
        hf_datasets.append(load_dataset(src["path"], **kwargs))

    # Calculate probability matrices
    weights = [src["weight"] for src in sources]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    print("Interleaving dataset arrays uniformly across proportional weights...")
    interleaved = interleave_datasets(
        hf_datasets,
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted",
    )

    # State containers
    buffer = []
    
    # RAM Optimization: Pre-allocate standard internal list to avoid structural memory overhead.
    # Appending massive sublists elements to python lists grows objects footprint dramatically.
    current_shard_tokens = []
    shard_counter = 0
    total_tokens_written = 0
    
    # Inline Dedup Lookback Cache
    seen_hashes = collections.deque(maxlen=DEDUP_WINDOW_SIZE)
    source_map = {src["path"]: src for src in sources}

    print("\n🚀 Pipeline fully operational. Streaming tokens...")
    print("-" * 60)

    for row in interleaved:
        # Fallback tracking resolution for wrapped streams
        src_config = source_map.get(row.get("__source__"), sources[0])
        
        text = _extract_text_by_config(row, src_config)
        if text is None or len(text) < 40:
            return None if text is None else len(text)
            continue
            
        # 1. Inline Window Deduplication
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in seen_hashes:
            continue
        seen_hashes.append(doc_hash)

        # 2. FastText Language Uniformity Classifier (CPU Bound)
        try:
            labels, scores = ft_model.predict(text.replace("\n", " "), k=1)
            label = labels[0].replace("__label__", "")
            if label != "en" or scores[0] < dcfg.lang_threshold:
                continue
        except Exception:
            continue

        # 3. Detoxify Content Guard Rail Filter (Dual GPU Round Robin Execution)
        try:
            # Alternates evaluation queries across both T4 GPUs dynamically on the fly
            active_detox_model = detox_models[gpu_selector]
            res = active_detox_model.predict(text)
            gpu_selector = (gpu_selector + 1) % 2 # Flip-flop index pointer between [0, 1]
            
            if res.get("toxicity", 0.0) >= dcfg.toxicity_threshold:
                continue
        except Exception:
            # Safely continue loop execution if string serialization fails
            gpu_selector = (gpu_selector + 1) % 2
            continue

        # 4. Tokenization and Document Bounds Constraints Verification
        tokens = sp.Encode(text)
        if not (dcfg.min_doc_tokens <= len(tokens) <= dcfg.max_doc_tokens):
            continue
            
        tokens.append(dcfg.eos_token_id)
        buffer.extend(tokens)

        # 5. Pack Streams Into Uniform Matrix Context Blocks
        while len(buffer) >= dcfg.seq_len:
            chunk = buffer[:dcfg.seq_len]
            current_shard_tokens.append(chunk)
            buffer = buffer[dcfg.seq_len:]
            total_tokens_written += dcfg.seq_len
            
            # 6. Periodic Batch Export to Gated Hub Destination Repo
            if len(current_shard_tokens) * dcfg.seq_len >= SHARD_SIZE_TOKENS:
                print(f"📦 Assembling Batch Shard {shard_counter} | Accumulated Target: {total_tokens_written:,} tokens")
                
                # Zero-copy reference construction into dynamic PyArrow arrays
                arr = np.array(current_shard_tokens, dtype=np.int32)
                ds = Dataset.from_dict({"input_ids": arr})
                
                # Pushes incrementally without localized storage overheads
                ds.push_to_hub(
                    hf_export_repo, 
                    split=f"train_shard_{shard_counter}",
                    private=True,
                    token=hf_token
                )
                
                # Reclaim memory space right away
                del arr
                del ds
                current_shard_tokens = []
                shard_counter += 1
                
        if total_tokens_written >= TARGET_TOKENS:
            print("-" * 60)
            print(f"🎉 Success! Targeted {total_tokens_written:,} tokens successfully baked & pushed.")
            break

if __name__ == "__main__":
    args = parse_args()
    build_and_upload_dataset(hf_export_repo=args.hf_export_repo)
