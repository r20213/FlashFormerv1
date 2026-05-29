import os
import sys
import numpy as np
import collections
import hashlib
from typing import Optional
from datasets import load_dataset, interleave_datasets, Dataset
import sentencepiece as spm
from detoxify import Detoxify
import fasttext

# Force path resolution for local configurations
sys.path.append(os.getcwd())
from config import DataConfig  # Ingests your exact architecture layouts

# --- LOCAL DEPLOYMENT TUNING PARAMETERS ---
TARGET_TOKENS = 20_000_000_000     # 20 Billion tokens total
SHARD_SIZE_TOKENS = 100_000_000   # 100M tokens per shard (~200 files total)
HF_EXPORT_REPO = "your-hf-username/prefiltered-packed-235m"
DEDUP_WINDOW_SIZE = 100_000       # Tracks last 100k document hashes seamlessly

print("Initializing DataConfig Specifications...")
dcfg = DataConfig()

# Explicitly override the legacy 'c++' directory key to prevent Hub routing crashes
for src in dcfg.sources:
    if src.get("path") == "bigcode/starcoderdata" and src.get("name") == "c++":
        src["name"] = "cpp"

print("Loading GPU-Accelerated Filter Handlers...")
detox_model = Detoxify("original", device="cuda")
ft_model = fasttext.load_model("tokenizer/fasttext_model.bin")
sp = spm.SentencePieceProcessor(model_file=dcfg.tokenizer_path)

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

def build_and_upload_dataset():
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
            continue
            
        # 1. Inline Window Deduplication
        doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if doc_hash in seen_hashes:
            continue
        seen_hashes.append(doc_hash)

        # 2. FastText Language Uniformity Classifier
        try:
            labels, scores = ft_model.predict(text.replace("\n", " "), k=1)
            label = labels[0].replace("__label__", "")
            if label != "en" or scores[0] < dcfg.lang_threshold:
                continue
        except:
            continue

        # 3. Detoxify Content Guard Rail Filter
        try:
            res = detox_model.predict(text)
            if res.get("toxicity", 0.0) >= dcfg.toxicity_threshold:
                continue
        except:
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
                
                arr = np.array(current_shard_tokens, dtype=np.int32)
                ds = Dataset.from_dict({"input_ids": arr})
                
                # Pushes incrementally without localized storage overheads
                ds.push_to_hub(
                    HF_EXPORT_REPO, 
                    split=f"train_shard_{shard_counter}",
                    private=True,
                    token=hf_token
                )
                
                current_shard_tokens = []
                shard_counter += 1
                
        if total_tokens_written >= TARGET_TOKENS:
            print("-" * 60)
            print(f"🎉 Success! Targeted {total_tokens_written:,} tokens successfully baked & pushed.")
            break

if __name__ == "__main__":
    build_and_upload_dataset()
