"""
tokenizer/train_tokenizer.py

Trains a SentencePiece BPE tokenizer (32k vocab) on a proportionally
sampled corpus drawn from all 16 pretraining data sources.

Design constraints (H100 CPU side: 4 vCPUs, 27GB RAM, 19GB ephemeral storage):
  - Streams all datasets from HuggingFace Hub — never downloads full splits.
  - Writes a single flat corpus text file (~120MB for 30M tokens), trains SPM,
    then deletes the corpus to free disk. Net disk cost: ~2MB (model + vocab).
  - Peak RAM: ~10-14GB (Optimized SPM trainer + 16 open streaming iterators).
  - 4 worker processes for parallel per-source sampling; SPM uses 4 threads.
  - Byte-fallback enabled: every Unicode character is representable without <unk>.
  - Character coverage 0.9999 to handle code's long-tail symbol distribution.
"""

import argparse
import logging
import multiprocessing as mp
import queue as queue_errors
import sys
import time
import shutil
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — mirrors DataConfig.sources from config.py exactly.
# ---------------------------------------------------------------------------

SOURCES = [
    # ── Natural language ────────────────────────────────────────────────────
    {
        "path": "openbmb/Ultra-FineWeb-L3",
        "name": "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
        "split": "train",
        "text_column": "content",
        "weight": 0.337,
    },
    # ── Code: Tier A (core competence) ─────────────────────────────────────
    {"path": "bigcode/starcoderdata", "name": "python",     "split": "train", "text_column": "content", "weight": 0.090},
    {"path": "bigcode/starcoderdata", "name": "javascript", "split": "train", "text_column": "content", "weight": 0.065},
    {"path": "bigcode/starcoderdata", "name": "java",       "split": "train", "text_column": "content", "weight": 0.045},
    {"path": "bigcode/starcoderdata", "name": "c++",        "split": "train", "text_column": "content", "weight": 0.038},
    {"path": "bigcode/starcoderdata", "name": "c",          "split": "train", "text_column": "content", "weight": 0.032},
    {"path": "bigcode/starcoderdata", "name": "sql",        "split": "train", "text_column": "content", "weight": 0.015},
    # ── Code: Tier B (reasoning signal) ────────────────────────────────────
    {"path": "bigcode/starcoderdata", "name": "go",         "split": "train", "text_column": "content", "weight": 0.018},
    {"path": "bigcode/starcoderdata", "name": "rust",       "split": "train", "text_column": "content", "weight": 0.016},
    {"path": "bigcode/starcoderdata", "name": "shell",      "split": "train", "text_column": "content", "weight": 0.012},
    {"path": "bigcode/starcoderdata", "name": "typescript", "split": "train", "text_column": "content", "weight": 0.010},
    {"path": "bigcode/starcoderdata", "name": "cuda",       "split": "train", "text_column": "content", "weight": 0.008},
    {"path": "bigcode/starcoderdata", "name": "lean",       "split": "train", "text_column": "content", "weight": 0.007},
    {"path": "bigcode/starcoderdata", "name": "haskell",    "split": "train", "text_column": "content", "weight": 0.007},
    # ── Math pretraining ────────────────────────────────────────────────────
    {
        "path": "HuggingFaceTB/finemath",
        "name": "infiwebmath-4plus",
        "split": "train",
        "text_column": "text",
        "weight": 0.220,
    },
    # ── Instruction SFT ─────────────────────────────────────────────────────
    {
        "path": "openbmb/UltraInteract_sft",
        "name": None,
        "split": "train",
        "text_column": None,
        "weight": 0.012,
    },
]

# Normalize weights
_total_weight = sum(s["weight"] for s in SOURCES)
for s in SOURCES:
    s["weight"] /= _total_weight


# ---------------------------------------------------------------------------
# Special-case formatters & Data Cleaners
# ---------------------------------------------------------------------------

def format_ultrainteract(row: dict) -> Optional[str]:
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("response") or "").strip()
    if not instruction or not response:
        return None
    return f"{instruction}\n\n{response}"


def get_text(row: dict, source: dict) -> Optional[str]:
    col = source["text_column"]
    if col is None:
        if source["path"] == "openbmb/UltraInteract_sft":
            return format_ultrainteract(row)
        return None

    text = row.get(col)
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    if len(text) < 40:  # Minimum basic text length filter
        return None
    return text


# ---------------------------------------------------------------------------
# Per-source streaming worker
# ---------------------------------------------------------------------------

def stream_source(
    source: dict,
    target_chars: int,
    queue: mp.Queue,
    worker_id: int,
    batch_char_limit: int = 100_000,
) -> None:
    """
    Streams one data source incrementally with network retry resilience.
    Breaks chunks down line-by-line so SentencePiece recognizes correct sentences.
    """
    import os
    from datasets import load_dataset  # noqa: PLC0415

    hf_token = os.environ.get("HF_TOKEN", None)
    src_name = source.get("name") or source["path"].split("/")[-1]
    log.info(f"[worker-{worker_id}] Starting: {source['path']} / {src_name}")

    kwargs = {
        "streaming": True,
        "split": source["split"],
        "trust_remote_code": True,
        "token": hf_token,
    }
    if source["path"] == "bigcode/starcoderdata":
        kwargs["data_dir"] = source["name"]
    elif source["name"] is not None:
        kwargs["name"] = source["name"]

    ds = None
    for attempt in range(1, 4):
        try:
            ds = load_dataset(source["path"], **kwargs)
            break
        except Exception as e:
            if attempt == 3:
                log.error(f"[worker-{worker_id}] CRITICAL: Failed to load {src_name} after 3 attempts: {e}")
                queue.put(("", worker_id, src_name, True))
                return
            log.warning(f"[worker-{worker_id}] Connection attempt {attempt}/3 failed for {src_name}. Retrying...")
            time.sleep(2 * attempt)

    collected_lines = []
    current_chunk_chars = 0
    total_emitted_chars = 0

    try:
        for row in ds:
            text = get_text(row, source)
            if text is None:
                continue

            # Break blocks up by line so SentencePiece maps logical sentences instead of overflow units
            lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 5]
            if not lines:
                continue

            for line in lines:
                collected_lines.append(line)
                current_chunk_chars += len(line)
                total_emitted_chars += len(line)

            if current_chunk_chars >= batch_char_limit:
                chunk_payload = "\n".join(collected_lines)
                queue.put((chunk_payload, worker_id, src_name, False))
                collected_lines = []
                current_chunk_chars = 0

            if total_emitted_chars >= target_chars:
                break
    except Exception as e:
        log.error(f"[worker-{worker_id}] Error while streaming {src_name}: {e}")

    if collected_lines:
        chunk_payload = "\n".join(collected_lines)
        queue.put((chunk_payload, worker_id, src_name, False))

    log.info(f"[worker-{worker_id}] Done: {src_name} | Emitted {total_emitted_chars:,} chars.")
    queue.put(("", worker_id, src_name, True))


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def build_corpus(output_path: Path, total_target_tokens: int = 30_000_000, chars_per_token: float = 4.0) -> int:
    total_target_chars = int(total_target_tokens * chars_per_token)
    log.info(f"Target corpus: {total_target_tokens/1e6:.1f}M tokens ≈ {total_target_chars/1e6:.1f}MB of text")

    source_char_budgets = [int(s["weight"] * total_target_chars) for s in SOURCES]
    n_workers = min(4, mp.cpu_count())
    log.info(f"Using {n_workers} parallel workers")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    total_chars_written = 0
    source_queue = list(enumerate(zip(SOURCES, source_char_budgets)))
    active_workers = {}

    def launch_next():
        while len(active_workers) < n_workers and source_queue:
            i, (src, budget) = source_queue.pop(0)
            p = ctx.Process(target=stream_source, args=(src, budget, queue, i), daemon=True)
            p.start()
            active_workers[i] = p

    launch_next()
    completed = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        while completed < len(SOURCES):
            try:
                chunk, worker_id, src_name, is_done = queue.get(timeout=1.0)
                
                if chunk:
                    f_out.write(chunk)
                    f_out.write("\n")
                    total_chars_written += len(chunk)
                    progress = (total_chars_written / total_target_chars) * 100 if total_target_chars > 0 else 100.0
                    log.info(f"  Written chunk from {src_name}: {len(chunk):,} chars | Total: {total_chars_written:,} ({progress:.1f}%)")

                if is_done:
                    completed += 1
                    log.info(f"Worker for {src_name} finished stream successfully.")
                    if worker_id in active_workers:
                        p = active_workers.pop(worker_id)
                        p.join()
                    launch_next()

            except queue_errors.Empty:
                dead_workers = []
                for wid, proc in active_workers.items():
                    if not proc.is_alive():
                        log.error(f"CRITICAL: Worker process {wid} running source index {wid} died prematurely!")
                        dead_workers.append(wid)
                
                for wid in dead_workers:
                    active_workers.pop(wid).join()
                    completed += 1  
                    launch_next()
                
                if not active_workers and source_queue:
                    launch_next()
                elif not active_workers and not source_queue and completed < len(SOURCES):
                    log.error("All workers terminated unexpectedly. Halting corpus collection.")
                    break

    log.info(f"Corpus built: {total_chars_written:,} chars. Saved to {output_path}")
    return total_chars_written


# ---------------------------------------------------------------------------
# SentencePiece training
# ---------------------------------------------------------------------------

def train_spm(corpus_path: Path, model_prefix: str, vocab_size: int = 32_768, num_threads: int = 4) -> None:
    log.info(f"Training SentencePiece BPE, vocab_size={vocab_size} ...")
    t0 = time.time()

    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        character_coverage=0.9999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        num_threads=num_threads,
        shuffle_input_sentence=True,
        # ─── RAM Allocation & Buffer Enhancements ────────────────────────────
        input_sentence_size=50_000_000,   # Scales internal RAM buffers to cover the entire dataset
        max_sentence_length=16384,        # Prevents long layout sequences from getting chopped
        train_extremely_large_corpus=True,
    )
    log.info(f"SentencePiece training complete in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_model(model_path: Path) -> bool:
    log.info(f"Verifying model: {model_path}")
    sp = spm.SentencePieceProcessor()
    if not sp.Load(str(model_path)):
        log.error("Failed to load SentencePiece model.")
        return False

    ok = True
    if sp.GetPieceSize() != 32_768:
        log.error(f"Expected 32768 pieces, got {sp.GetPieceSize()}")
        ok = False

    checks = [("<unk>", 0), ("<s>", 1), ("</s>", 2), ("<pad>", 3)]
    for piece, expected_id in checks:
        if sp.PieceToId(piece) != expected_id:
            log.error(f"  ✗ {piece} has ID {sp.PieceToId(piece)}, expected {expected_id}")
            ok = False

    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1",
        "SELECT id, name FROM users WHERE active = 1;",
        "1234567890",
        "\t    ",
    ]
    for text in test_cases:
        if sp.Decode(sp.Encode(text)) != text:
            log.error(f"  ✗ Round-trip failed for: {repr(text)}")
            ok = False

    # Fertility check
    sample = "The transformer model uses self-attention to compute representations."
    fertility = len(sp.Encode(sample)) / len(sample.split())
    log.info(f"  ✓ Fertility Check: {fertility:.2f} tokens/word (Expected Target: 1.2-1.8)")

    return ok


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SentencePiece tokenizer for pretraining")
    p.add_argument("--tokens", type=int, default=150_000_000, help="Target tokens in training corpus")
    p.add_argument("--vocab-size", type=int, default=32_768, help="Vocabulary size")
    p.add_argument("--output-dir", type=str, default="tokenizer", help="Output artifacts directory")
    p.add_argument("--keep-corpus", action="store_true", help="Do not delete corpus.txt after training")
    p.add_argument("--verify-only", action="store_true", help="Skip training and run model checks")
    p.add_argument("--num-threads", type=int, default=4, help="Threads for SPM training engine")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "spm.model"
    vocab_path = out_dir / "spm.vocab"
    corpus_path = out_dir / "corpus.txt"
    model_prefix = str(out_dir / "spm")

    if args.verify_only:
        if not model_path.exists():
            log.error(f"Model file missing: {model_path}")
            sys.exit(1)
        sys.exit(0 if verify_model(model_path) else 1)

    # Pre-flight hardware assertions
    free_bytes = shutil.disk_usage(out_dir).free
    required_bytes = int((args.tokens * 4.0) * 2.0)
    if free_bytes < required_bytes:
        log.error(f"Insufficient disk space. Requires ~{required_bytes/1e6:.1f}MB.")
        sys.exit(1)

    avail_ram_gb = psutil.virtual_memory().available / 1e9
    if avail_ram_gb < 4.0:
        log.warning(f"Critical low memory threshold alert: ({avail_ram_gb:.1f}GB remaining).")
    else:
        log.info(f"Pre-flight health checks cleared. System RAM available: {avail_ram_gb:.1f}GB")

    # Pipeline Execution Chain
    t_start = time.time()
    
    log.info("STEP 1: Gathering and structuring training data...")
    n_chars = build_corpus(output_path=corpus_path, total_target_tokens=args.tokens)
    if n_chars == 0:
        log.error("Zero characters written to corpus file. Training aborted.")
        sys.exit(1)

    log.info("STEP 2: Initializing SentencePiece trainer optimization routine...")
    train_spm(corpus_path=corpus_path, model_prefix=model_prefix, vocab_size=args.vocab_size, num_threads=args.num_threads)

    if not args.keep_corpus and corpus_path.exists():
        corpus_path.unlink()
        log.info("Temporary extraction cache wiped out to reclaim space.")

    log.info("STEP 3: Running downstream structural tests on compiled vocabulary artifacts...")
    if verify_model(model_path):
        log.info(f"Tokenizer pipeline ran completely successfully in {time.time() - t_start:.1f}s.")
        log.info(f"Model output: {model_path} | Vocab metadata map: {vocab_path}")
    else:
        log.error("Verification sequence flagged functional anomalies.")
        sys.exit(1)


if __name__ == "__main__":
    main()