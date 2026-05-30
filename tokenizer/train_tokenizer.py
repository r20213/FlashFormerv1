"""
train_tokenizer.py
──────────────────
Trains a 16K Rust-based BPE tokenizer on 150M tokens from
Nemotron-CC-Math-v1 and DeepMath-103K, evaluates it against
reference tokenizers on a held-out 10M-token set and a
deterministic stress-test suite, then pushes to HF Hub.

Hardware target : 4 vCPUs, 27 GB RAM (Modal CPU worker)
Runtime estimate: ~1.5–2 hours end-to-end

Usage
-----
    HF_TOKEN=hf_xxx python train_tokenizer.py

Environment variables
---------------------
    HF_TOKEN   (required) — HuggingFace write-access token
"""

from __future__ import annotations

import gc
import os
import re
import sys
import json
import time
import random
import unicodedata
import logging
import contextlib
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List, Dict, Tuple, Optional

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Strip, Replace, Sequence as NormSequence
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

import config as C

# ─────────────────────────────────────────────────────────────
# 0. Bootstrap
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mathformer")


def _require_env(var: str) -> str:
    val = os.environ.get(var, "").strip()
    if not val:
        log.error(
            f"Required environment variable '{var}' is not set.\n"
            f"Export it before running:  export {var}=<your_token>"
        )
        sys.exit(1)
    return val


HF_TOKEN = _require_env("HF_TOKEN")

# ── Resolve paths: prefer Modal volume mounts when present ───
_on_modal    = Path(C.MODAL_VOLUME_MOUNT).exists()
DATA_DIR     = C.MODAL_DATA_DIR     if _on_modal else C.DATA_DIR
TOK_DIR      = C.MODAL_TOK_DIR      if _on_modal else C.TOKENIZER_DIR
CORPUS_TRAIN = C.MODAL_CORPUS_TRAIN if _on_modal else C.CORPUS_TRAIN_PATH
CORPUS_EVAL  = C.MODAL_CORPUS_EVAL  if _on_modal else C.CORPUS_EVAL_PATH
TOK_JSON     = C.MODAL_TOK_JSON     if _on_modal else C.TOKENIZER_JSON_PATH

DATA_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(C.RANDOM_SEED)

# ── Corpus size cap ──────────────────────────────────────────
# 120M whitespace-words ≈ 150M real BPE tokens after fragmentation.
# Keeps the corpus file under ~1.5 GB so BPE trainer stays in RAM.
TRAIN_WORD_CAP = 120_000_000


# ─────────────────────────────────────────────────────────────
# 1. Timer utility
# ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def timer(label: str):
    """Wall-clock timer. Logs start, end, and elapsed H:MM:SS."""
    log.info(f"⏱  [{label}] starting …")
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        td = timedelta(seconds=int(elapsed))
        log.info(f"✓  [{label}] done in {td} ({elapsed:.1f}s)")


def _log_memory(label: str = "") -> None:
    try:
        import psutil
        proc  = psutil.Process()
        rss   = proc.memory_info().rss / 1024**3
        avail = psutil.virtual_memory().available / 1024**3
        tag   = f" [{label}]" if label else ""
        log.info(f"  RAM{tag}: {rss:.2f}GB RSS used, {avail:.2f}GB available")
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────
# 2. Text normalisation
# ─────────────────────────────────────────────────────────────

_UNICODE_SPACES = re.compile(
    r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\ufeff]"
)
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Normalisation pipeline (order is deliberate):
    1. NFC  — compose combining characters. NOT NFKC (preserves LaTeX forms).
    2. Replace exotic Unicode spaces with ASCII 0x20.
    3. Collapse 3+ consecutive blank lines to exactly 2.
    4. Strip leading/trailing whitespace.
    No lowercasing — math is case-sensitive (x ≠ X).
    """
    if not text:
        return ""
    text = unicodedata.normalize(C.UNICODE_NORM_FORM, text)
    text = _UNICODE_SPACES.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def build_deepmath_text(row: dict) -> str:
    parts = [row.get(C.DEEPMATH_TEXT_COL, "")]
    for col in C.DEEPMATH_AUX_COLS:
        val = row.get(col, "") or ""
        if val.strip():
            parts.append(val.strip())
    return "\n\n".join(p for p in parts if p.strip())


def _approx_token_count(text: str) -> int:
    return len(text.split())


# ─────────────────────────────────────────────────────────────
# 3. Dataset streamers
# ─────────────────────────────────────────────────────────────

def stream_nemotron(
    n_rows: int,
    skip_indices: Optional[set] = None,
) -> Iterator[str]:
    log.info(f"Streaming {n_rows:,} rows from Nemotron …")
    ds = load_dataset(
        C.NEMOTRON_DATASET,
        C.NEMOTRON_SUBSET,
        split=C.NEMOTRON_SPLIT,
        streaming=True,
        token=HF_TOKEN,
    )
    collected = 0
    for idx, row in enumerate(ds):
        if skip_indices and idx in skip_indices:
            continue
        if collected >= n_rows:
            break
        text = normalize_text(row.get(C.NEMOTRON_TEXT_COL, "") or "")
        if len(text) > 50:
            yield text
            collected += 1
    log.info(f"  → yielded {collected:,} Nemotron rows")


def stream_deepmath(
    n_rows: int,
    skip_indices: Optional[set] = None,
) -> Iterator[str]:
    """
    DeepMath is only 103K rows — small enough to load fully, but we
    use streaming=True to avoid the Arrow in-memory overhead that
    caused the OOM in the previous run.
    """
    log.info(f"Streaming DeepMath ({n_rows:,} rows requested) …")
    ds = load_dataset(
        C.DEEPMATH_DATASET,
        split=C.DEEPMATH_SPLIT,
        streaming=True,
        token=HF_TOKEN,
    )
    # Streaming datasets support buffer-based shuffle
    ds = ds.shuffle(seed=C.RANDOM_SEED, buffer_size=10_000)
    collected = 0
    for idx, row in enumerate(ds):
        if skip_indices and idx in skip_indices:
            continue
        if collected >= n_rows:
            break
        text = normalize_text(build_deepmath_text(row))
        if len(text) > 50:
            yield text
            collected += 1
    log.info(f"  → yielded {collected:,} DeepMath rows")


# ─────────────────────────────────────────────────────────────
# 4. Corpus file construction
# ─────────────────────────────────────────────────────────────

def build_corpus_files() -> None:
    """
    Write CORPUS_TRAIN and CORPUS_EVAL (one document per line).

    Key changes vs v1:
    - TRAIN_WORD_CAP hard-stops corpus at ~120M words (~1.5GB file)
      so BPE trainer never exceeds available RAM.
    - DeepMath now uses streaming=True (see stream_deepmath) to avoid
      loading the full Arrow table into memory.
    - File-size warning fires at >2GB.
    """
    if CORPUS_TRAIN.exists() and CORPUS_EVAL.exists():
        log.info("Corpus files already exist — skipping rebuild.")
        return

    log.info("═" * 60)
    log.info("Building corpus files …")
    log.info("═" * 60)
    _log_memory("corpus-start")

    # Eval rows come from the first N indices — training skips these.
    nemotron_eval_skip = set(range(C.NEMOTRON_EVAL_ROWS))
    deepmath_eval_skip = set(range(C.DEEPMATH_EVAL_ROWS))

    # ── Eval corpus ──────────────────────────────────────────
    log.info("Writing eval corpus …")
    eval_tokens = 0
    with open(CORPUS_EVAL, "w", encoding="utf-8") as f:
        for text in stream_nemotron(C.NEMOTRON_EVAL_ROWS, skip_indices=None):
            f.write(text + "\n")
            eval_tokens += _approx_token_count(text)
            if eval_tokens >= C.EVAL_TOKEN_BUDGET:
                break
        for text in stream_deepmath(C.DEEPMATH_EVAL_ROWS, skip_indices=None):
            f.write(text + "\n")
            eval_tokens += _approx_token_count(text)
    log.info(f"  Eval corpus: ~{eval_tokens:,} whitespace-words")

    # ── Train corpus ─────────────────────────────────────────
    log.info(f"Writing train corpus (cap: {TRAIN_WORD_CAP:,} words) …")
    train_tokens = 0

    with open(CORPUS_TRAIN, "w", encoding="utf-8") as f:

        for text in stream_nemotron(
            C.NEMOTRON_TRAIN_ROWS,
            skip_indices=nemotron_eval_skip,
        ):
            f.write(text + "\n")
            train_tokens += _approx_token_count(text)
            if train_tokens >= TRAIN_WORD_CAP:
                log.info("  Train word cap reached during Nemotron stream.")
                break

        if train_tokens < TRAIN_WORD_CAP:
            for text in stream_deepmath(
                C.DEEPMATH_TRAIN_ROWS,
                skip_indices=deepmath_eval_skip,
            ):
                f.write(text + "\n")
                train_tokens += _approx_token_count(text)
                if train_tokens >= TRAIN_WORD_CAP:
                    log.info("  Train word cap reached during DeepMath stream.")
                    break

    size_gb = os.path.getsize(CORPUS_TRAIN) / 1024**3
    log.info(f"  Train corpus: ~{train_tokens:,} whitespace-words | {size_gb:.2f} GB")

    if size_gb > 2.0:
        log.warning(
            f"Corpus is {size_gb:.2f} GB — high OOM risk during BPE training. "
            f"Reduce TRAIN_WORD_CAP in train_tokenizer.py and re-run."
        )

    _log_memory("corpus-end")
    log.info("Corpus files ready.")


# ─────────────────────────────────────────────────────────────
# 5. Tokenizer construction & training
# ─────────────────────────────────────────────────────────────

def build_tokenizer() -> Tokenizer:
    """
    Normaliser  : NFC → Strip → exotic-space replacements
    Pre-tokeniser: Regex Split (isolated) using config.PRETOKENIZER_REGEX

    Regex alternation order guarantees:
      1. Whitespace  → always its own isolated piece
      2. LaTeX cmds  → \frac, \alpha, etc. as one piece
      3. Digits      → one piece per digit (123 → ['1','2','3'])
      4. Alpha runs  → no digit contamination
      5. Operators   → one piece each
      6. Brackets    → one piece each
      7. Fallthrough → any remaining Unicode char

    BPE can only merge within pre-token boundaries, so spaces never
    attach to words and digits never attach to letters.
    """
    tokenizer = Tokenizer(BPE(unk_token=C.UNK_TOKEN))

    tokenizer.normalizer = NormSequence([
        NFC(),
        Strip(),
        Replace("\u00a0", " "),   # non-breaking space
        Replace("\u2009", " "),   # thin space
        Replace("\u202f", " "),   # narrow no-break space
        Replace("\u2003", " "),   # em space
        Replace("\u2002", " "),   # en space
        Replace("\u200b", ""),    # zero-width space → remove
        Replace("\ufeff", ""),    # BOM → remove
    ])

    tokenizer.pre_tokenizer = Split(
        pattern=C.PRETOKENIZER_REGEX,
        behavior="isolated",
    )

    return tokenizer


def train_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    # RAYON_NUM_THREADS must be set before the Rust thread pool
    # initialises — set it here as a safety net (shell export is primary).
    os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 4))
    log.info(f"RAYON_NUM_THREADS = {os.environ['RAYON_NUM_THREADS']}")

    trainer = BpeTrainer(
        vocab_size=C.VOCAB_SIZE,
        min_frequency=C.MIN_FREQUENCY,
        special_tokens=C.SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=[],
        continuing_subword_prefix="##",
        end_of_word_suffix="",
    )

    log.info("Starting BPE training …")
    tokenizer.train(files=[str(CORPUS_TRAIN)], trainer=trainer)
    log.info(f"Final vocab size: {tokenizer.get_vocab_size():,}")
    return tokenizer


def wrap_as_fast_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=C.UNK_TOKEN,
        bos_token=C.BOS_TOKEN,
        eos_token=C.EOS_TOKEN,
        pad_token=C.PAD_TOKEN,
        model_max_length=4096,
        padding_side="right",
        truncation_side="right",
    )
    fast.add_special_tokens({
        "additional_special_tokens": [
            t for t in C.SPECIAL_TOKENS
            if t not in (C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN, C.PAD_TOKEN)
        ]
    })
    return fast


# ─────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────

def _encode(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)
        return result if isinstance(result, list) else result.ids
    return tokenizer(text)["input_ids"]


def corpus_metrics(
    tokenizer,
    name: str,
    corpus_path: Path,
    max_docs: int = 50_000,
) -> Dict[str, float]:
    log.info(f"  Corpus metrics [{name}] …")

    total_words = total_tokens = unk_count = continued = 0
    math_ops_seen: set = set()
    math_ops_as_single: set = set()

    unk_id = None
    try:
        unk_id = (tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}).get(C.UNK_TOKEN)
    except Exception:
        pass

    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            words = line.split()
            ids   = _encode(tokenizer, line)
            total_words  += len(words)
            total_tokens += len(ids)
            if unk_id is not None:
                unk_count += ids.count(unk_id)
            try:
                conv = tokenizer.convert_ids_to_tokens(ids)
                continued += sum(1 for t in conv if t and t.startswith("##"))
            except Exception:
                pass
            for op in C.MATH_OPERATORS_CORPUS:
                if op in line:
                    math_ops_seen.add(op)
                    try:
                        op_ids = _encode(tokenizer, op)
                        if len(op_ids) == 1 and op_ids[0] != unk_id:
                            math_ops_as_single.add(op)
                    except Exception:
                        pass

    fertility = total_tokens / max(total_words, 1)
    pcw       = continued   / max(total_tokens, 1)
    unk_rate  = unk_count   / max(total_tokens, 1)
    math_cov  = (len(math_ops_as_single) / max(len(math_ops_seen), 1)
                 if math_ops_seen else float("nan"))

    return {
        "fertility":             round(fertility,       4),
        "continued_word_pct":   round(pcw      * 100,  2),
        "unk_rate":              round(unk_rate * 100,  4),
        "math_symbol_coverage": round(math_cov * 100,  2),
        "vocab_size":           (tokenizer.get_vocab_size()
                                  if hasattr(tokenizer, "get_vocab_size")
                                  else len(tokenizer.get_vocab())),
    }


def stress_test(tokenizer, name: str) -> Dict[str, float]:
    log.info(f"  Stress-test [{name}] …")

    try:
        unk_id = tokenizer.get_vocab().get(C.UNK_TOKEN)
    except Exception:
        unk_id = None

    # Digit integrity
    digit_pass = sum(
        1 for d in C.PROBE_DIGITS
        if len(_encode(tokenizer, d)) == 1 and _encode(tokenizer, d)[0] != unk_id
    )
    digit_integrity = digit_pass / len(C.PROBE_DIGITS)

    # Arithmetic readiness + PCW for numbers
    tok_counts = [len(_encode(tokenizer, n)) for n in C.PROBE_NUMBERS]
    arith_readiness = float(np.mean(tok_counts))
    multi = [n for n in C.PROBE_NUMBERS if sum(c.isdigit() for c in n) > 1]
    pcw_numbers = sum(1 for n in multi if len(_encode(tokenizer, n)) > 1) / max(len(multi), 1)

    # Single-token retention rate — LaTeX commands
    strr_pass = sum(
        1 for cmd in C.PROBE_LATEX_COMMANDS
        if len(_encode(tokenizer, cmd)) == 1 and _encode(tokenizer, cmd)[0] != unk_id
    )
    strr = strr_pass / len(C.PROBE_LATEX_COMMANDS)

    # Chars per token on math expressions
    total_chars = sum(len(e.replace(" ", "")) for e in C.PROBE_EXPRESSIONS)
    total_etoks = sum(len(_encode(tokenizer, e)) for e in C.PROBE_EXPRESSIONS)
    cpt = total_chars / max(total_etoks, 1)

    return {
        "digit_integrity_rate":           round(digit_integrity * 100, 2),
        "arithmetic_readiness_mean_toks": round(arith_readiness,       3),
        "strr_latex_pct":                 round(strr            * 100, 2),
        "pcw_numbers_fragmented_pct":     round(pcw_numbers     * 100, 2),
        "cpt_expressions":                round(cpt,                   3),
    }


def run_full_evaluation(
    our_tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Dict, Dict]:
    """
    Evaluate MathFormer then each reference tokenizer.
    Each reference is loaded, evaluated, then explicitly freed before
    the next one loads — prevents cumulative RAM growth that caused OOM.
    """
    from transformers import AutoTokenizer

    log.info("═" * 60)
    log.info("Running full evaluation …")
    log.info("═" * 60)

    all_corpus: Dict = {}
    all_stress: Dict = {}

    # ── MathFormer (already in memory) ───────────────────────
    with timer("Eval MathFormer"):
        all_corpus["MathFormer"] = corpus_metrics(
            our_tokenizer.backend_tokenizer, "MathFormer", CORPUS_EVAL
        )
        all_stress["MathFormer"] = stress_test(
            our_tokenizer.backend_tokenizer, "MathFormer"
        )

    # Free our tokenizer before loading large reference models
    del our_tokenizer
    gc.collect()
    _log_memory("after-MathFormer-eval")

    # ── Reference tokenizers — one at a time ─────────────────
    for display_name, hf_id in C.REFERENCE_TOKENIZERS:
        with timer(f"Eval {display_name}"):
            ref_tok = backend = None
            try:
                log.info(f"  Loading {hf_id} …")
                ref_tok = AutoTokenizer.from_pretrained(
                    hf_id, token=HF_TOKEN, trust_remote_code=True
                )
                backend = getattr(ref_tok, "_tokenizer", None) or ref_tok
                all_corpus[display_name] = corpus_metrics(
                    backend, display_name, CORPUS_EVAL
                )
                all_stress[display_name] = stress_test(backend, display_name)
            except Exception as e:
                log.warning(f"  Could not load {hf_id}: {e}")
                all_corpus[display_name] = {"error": str(e)}
                all_stress[display_name] = {"error": str(e)}
            finally:
                # Always free — even on exception
                del ref_tok, backend
                gc.collect()
                _log_memory(f"after-{display_name}")

    return all_corpus, all_stress


def print_results_table(
    corpus_results: Dict[str, Dict],
    stress_results: Dict[str, Dict],
) -> None:
    CORPUS_COLS = [
        ("fertility",             "Fertility↓"),
        ("continued_word_pct",   "Cont.Word%↓"),
        ("unk_rate",              "UNK Rate↓"),
        ("math_symbol_coverage", "MathSym%↑"),
        ("vocab_size",            "Vocab"),
    ]
    STRESS_COLS = [
        ("digit_integrity_rate",          "Digit Integrity%↑"),
        ("arithmetic_readiness_mean_toks","ArithReady(toks)↓"),
        ("strr_latex_pct",                "STRR LaTeX%↑"),
        ("pcw_numbers_fragmented_pct",    "NumFrag%↑"),
        ("cpt_expressions",               "CPT Expr↑"),
    ]

    def _fmt(v) -> str:
        if isinstance(v, float): return f"{v:.3f}"
        if isinstance(v, int):   return f"{v:,}"
        return str(v)

    def _table(results, cols, title):
        header = f"{'Model':<26}" + "".join(f"{h:<22}" for _, h in cols)
        print(f"\n{'═'*90}")
        print(f"  {title}")
        print("═" * 90)
        print(header)
        print("─" * 90)
        for name, row in results.items():
            line = f"{name:<26}"
            for key, _ in cols:
                line += f"{_fmt(row.get(key, '—')):<22}"
            print(line)
        print("─" * 90)

    _table(corpus_results, CORPUS_COLS, "A. Corpus Metrics (held-out eval set)")
    _table(stress_results, STRESS_COLS, "B. Stress-Test Suite (deterministic probes)")
    print()


def save_results(corpus_results: Dict, stress_results: Dict) -> None:
    path = TOK_DIR / "evaluation_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"corpus_metrics": corpus_results, "stress_test": stress_results},
            f, indent=2, ensure_ascii=False,
        )
    log.info(f"Evaluation results → {path}")


# ─────────────────────────────────────────────────────────────
# 7. Hub push
# ─────────────────────────────────────────────────────────────

def push_to_hub(fast_tokenizer: PreTrainedTokenizerFast) -> None:
    log.info(f"Pushing to Hub: {C.HUB_REPO_ID} …")
    fast_tokenizer.push_to_hub(
        C.HUB_REPO_ID,
        commit_message=C.HUB_COMMIT_MESSAGE,
        token=HF_TOKEN,
        private=False,
    )
    log.info("Hub push complete.")


# ─────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    wall_start = time.time()

    log.info("╔══════════════════════════════════════╗")
    log.info("║   MathFormer Tokenizer Training      ║")
    log.info("╚══════════════════════════════════════╝")

    # ── Memory guard ─────────────────────────────────────────
    try:
        import psutil
        vm    = psutil.virtual_memory()
        avail = vm.available / 1024**3
        total = vm.total     / 1024**3
        log.info(f"RAM at startup: {avail:.1f}GB available / {total:.1f}GB total")
        if avail < 4.0:
            log.error("< 4GB available — high OOM risk. Free memory and retry.")
            sys.exit(1)
    except ImportError:
        log.warning("psutil not installed — skipping memory check. pip install psutil")

    # ── HF datasets: never cache to RAM ──────────────────────
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"

    log.info(f"Modal volume detected : {_on_modal}")
    log.info(f"Train corpus path     : {CORPUS_TRAIN}")
    log.info(f"Eval  corpus path     : {CORPUS_EVAL}")
    log.info(f"Tokenizer output      : {TOK_DIR}")
    log.info(f"Train word cap        : {TRAIN_WORD_CAP:,}")

    with timer("Corpus build"):
        build_corpus_files()

    with timer("Tokenizer init"):
        tokenizer = build_tokenizer()

    with timer("BPE training"):
        tokenizer = train_tokenizer(tokenizer)

    _log_memory("post-training")

    with timer("Save tokenizer"):
        tokenizer.save(str(TOK_JSON))
        log.info(f"Raw tokenizer JSON → {TOK_JSON}")
        fast_tok = wrap_as_fast_tokenizer(tokenizer)
        fast_tok.save_pretrained(str(TOK_DIR))
        log.info(f"Fast tokenizer     → {TOK_DIR}")

    with timer("Evaluation"):
        corpus_res, stress_res = run_full_evaluation(fast_tok)
        print_results_table(corpus_res, stress_res)
        save_results(corpus_res, stress_res)

    # ── Sanity gates before push ──────────────────────────────
    unk_rate = corpus_res.get("MathFormer", {}).get("unk_rate", 99.0)
    if unk_rate > 0.5:
        log.warning(
            f"UNK rate {unk_rate:.2f}% exceeds 0.5% threshold. "
            f"Review corpus or increase vocab size before deploying."
        )

    dig = stress_res.get("MathFormer", {}).get("digit_integrity_rate", 0.0)
    if dig < 100.0:
        log.warning(
            f"Digit integrity {dig:.1f}% < 100% target. "
            f"Review PRETOKENIZER_REGEX in config.py."
        )

    with timer("Hub push"):
        push_to_hub(fast_tok)

    total_td = timedelta(seconds=int(time.time() - wall_start))
    log.info("═" * 50)
    log.info(f"  Total wall time: {total_td}")
    log.info("═" * 50)


if __name__ == "__main__":
    main()