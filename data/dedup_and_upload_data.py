import os
import sys
import argparse
import numpy as np
import collections
import hashlib
from typing import Optional
from pathlib import Path

# Core imports
from datasets import load_dataset, interleave_datasets, Dataset
import sentencepiece as spm

# Performance Tuning for 4 vCPU environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Adds parent directory to your search path
parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
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
    parser = argparse.ArgumentParser(description="High-Throughput Dataset Tokenization and Pack Pipeline")
    parser.add_argument(
        "--hf_export_repo", 
        type=str, 
        required=True, 
        help="Mandatory Hugging Face repository destination (e.g., 'username/prefiltered-packed-235m')"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        required=True, 
        help="Mandatory path to the SentencePiece tokenizer model file (e.g., 'tokenizer.model')"
    )
    return parser.parse_args()

def _extract_text_by_config(row: dict, src_config: dict) -> Optional[str]:
    """Extracts text strings adhering exactly to DataConfig structural keys."""
    path = src_config["path"]

    if path == "openbmb/UltraInteract_sft":
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        if not instruction or not response:
            return None
        return f"{instruction}\n\n{response}"

    col = "content" if path == "bigcode/starcoderdata" else src_config.get("text_column")
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

    print("Loading Tokenizer Processor...")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    sources = dcfg.sources
    hf_token = os.environ.get("HF_TOKEN", None)
    hf_datasets = []
    
    print("Binding streaming generators to HuggingFace Hub repositories...")
    for idx, src in enumerate(sources):
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
            
        ds = load_dataset(src["path"], **kwargs)
        
        # Resolve target parsing columns for this stream
        text_col = "content" if src["path"] == "bigcode/starcoderdata" else src.get("text_column")
        source_key = f"{src['path']}:{src.get('name', '')}"
        
        # 1. Map to resolve the source key tracker
        ds = ds.map(lambda r, sk=source_key: {"__resolved_source__": sk})
        
        # 2. Strict Column Selection: Isolates structural keys and drops dirty meta configurations
        columns_to_keep = ["__resolved_source__"]
        if text_col:
            columns_to_keep.append(text_col)
        if src["path"] == "openbmb/UltraInteract_sft":
            columns_to_keep.extend(["instruction", "response"])
            
        # Extract features safely to prune conflicting float/int types
        all_cols = list(ds.features.keys()) if hasattr(ds, "features") and ds.features else []
        if all_cols:
            cols_to_remove = [c for c in all_cols if c not in columns_to_keep]
            ds = ds.remove_columns(cols_to_remove)
            
        hf_datasets.append(ds)

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
    current_shard_tokens = []
    shard_counter = 0
    total_tokens_written = 0
    
    # Inline Dedup Lookback Cache
    seen_hashes = collections.deque(maxlen=DEDUP_WINDOW_SIZE)
    source_map = {f"{src['path']}:{src.get('name', '')}": src for src in sources}

    print("\n🚀 Pipeline fully operational. Streaming tokens...")
    print("-" * 60)

    for row in interleaved:
        # Fallback tracking resolution for wrapped streams
        resolved_key = row.get("__resolved_source__")
        src_config = source_map.get(resolved_key, sources[0])
        
        text = _extract_text_by_config(row, src_config)
        
        if text is None or len(text) < 40:
            continue
            
        # 1. Inline Window Deduplication
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in seen_hashes:
            continue
        seen_hashes.append(doc_hash)

        # 2. Tokenization and Document Bounds Constraints Verification
        tokens = sp.Encode(text)
        if not (dcfg.min_doc_tokens <= len(tokens) <= dcfg.max_doc_tokens):
            continue
            
        tokens.append(dcfg.eos_token_id)
        buffer.extend(tokens)

        # 3. Pack Streams Into Uniform Matrix Context Blocks
        while len(buffer) >= dcfg.seq_len:
            chunk = buffer[:dcfg.seq_len]
            current_shard_tokens.append(chunk)
            buffer = buffer[dcfg.seq_len:]
            total_tokens_written += dcfg.seq_len
            
            # 4. Periodic Batch Export to Gated Hub Destination Repo
            if len(current_shard_tokens) * dcfg.seq_len >= SHARD_SIZE_TOKENS:
                print(f"📦 Assembling Batch Shard {shard_counter} | Accumulated Target: {total_tokens_written:,} tokens")
                
                # Zero-copy reference construction into dynamic PyArrow arrays
                arr = np.array(current_shard_tokens, dtype=np.int32)
                ds_export = Dataset.from_dict({"input_ids": arr})
                
                # Pushes incrementally without localized storage overheads
                ds_export.push_to_hub(
                    hf_export_repo, 
                    split=f"train_shard_{shard_counter}",
                    private=True,
                    token=hf_token
                )
                
                # Reclaim memory space right away
                del arr
                del ds_export
                current_shard_tokens = []
                shard_counter += 1
                
        if total_tokens_written >= TARGET_TOKENS:
            print("-" * 60)
            print(f"🎉 Success! Targeted {total_tokens_written:,} tokens successfully baked & pushed.")
            break

if __name__ == "__main__":
    args = parse_args()
    build_and_upload_dataset(hf_export_repo=args.hf_export_repo)