"""
train_tokenizer.py
──────────────────
Optimized, low-RAM BPE tokenizer trainer for MathFormer.
Feeds data directly via memory-efficient iterators to keep RSS low.
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
# 0. Setup & Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mathformer")

# 👇 SILENCE THE HTTP/DATASETS CHATTER HERE 👇
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)

def _require_env(var: str) -> str:
    val = os.environ.get(var, "").strip()
    if not val:
        log.error(f"Required environment variable '{var}' is missing.")
        sys.exit(1)
    return val

HF_TOKEN = _require_env("HF_TOKEN")

# Resolve Paths
_on_modal = Path(C.MODAL_VOLUME_MOUNT).exists()
TOK_DIR = C.MODAL_TOK_DIR if _on_modal else C.TOKENIZER_DIR
TOK_JSON = C.MODAL_TOK_JSON if _on_modal else C.TOKENIZER_JSON_PATH
TOK_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Utilities
# ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def timer(label: str):
    log.info(f"⏱  [{label}] starting …")
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        log.info(f"✓  [{label}] done in {timedelta(seconds=int(elapsed))} ({elapsed:.1f}s)")

def _log_memory(label: str = "") -> None:
    try:
        import psutil
        proc = psutil.Process()
        rss = proc.memory_info().rss / 1024**3
        avail = psutil.virtual_memory().available / 1024**3
        log.info(f"  RAM [{label}]: {rss:.2f}GB RSS used, {avail:.2f}GB available")
    except ImportError:
        pass

# ─────────────────────────────────────────────────────────────
# 2. Normalization & Text Processing
# ─────────────────────────────────────────────────────────────

_UNICODE_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\ufeff]")
_MULTI_BLANK = re.compile(r"\n{3,}")

def normalize_text(text: str) -> str:
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

# ─────────────────────────────────────────────────────────────
# 3. Low-RAM Direct Iterators
# ─────────────────────────────────────────────────────────────

class TrainingCorpusIterator:
    """Streamed generator that groups strings into small batches to prevent Rust thread queue inflation."""
    def __init__(self, token: str, word_cap: int, batch_size: int = 1_000):
        self.token = token
        self.word_cap = word_cap
        self.batch_size = batch_size
        self.word_count = 0

    def __iter__(self) -> Iterator[List[str]]:
        self.word_count = 0
        eval_nemotron_offset = C.NEMOTRON_EVAL_ROWS
        eval_deepmath_offset = C.DEEPMATH_EVAL_ROWS
        current_batch = []

        # 1. Stream Nemotron
        nemotron_ds = load_dataset(
            C.NEMOTRON_DATASET, C.NEMOTRON_SUBSET,
            split=C.NEMOTRON_SPLIT, streaming=True, token=self.token
        ).skip(eval_nemotron_offset)

        for row in nemotron_ds:
            if self.word_count >= self.word_cap:
                if current_batch: yield current_batch
                return
            text = normalize_text(row.get(C.NEMOTRON_TEXT_COL, "") or "")
            if len(text) > 50:
                self.word_count += len(text.split())
                current_batch.append(text)
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []

        # 2. Stream DeepMath (Lowered buffer size to 2,000 for tight RAM stability)
        deepmath_ds = load_dataset(
            C.DEEPMATH_DATASET, split=C.DEEPMATH_SPLIT,
            streaming=True, token=self.token
        ).skip(eval_deepmath_offset).shuffle(seed=C.RANDOM_SEED, buffer_size=2_000)

        for row in deepmath_ds:
            if self.word_count >= self.word_cap:
                if current_batch: yield current_batch
                return
            text = normalize_text(build_deepmath_text(row))
            if len(text) > 50:
                self.word_count += len(text.split())
                current_batch.append(text)
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []

        if current_batch:
            yield current_batch


def get_eval_set(token: str) -> List[str]:
    """Loads a strict, downsized reference eval list to avoid anchoring RAM."""
    log.info("Collecting static evaluation slice...")
    eval_data = []
    
    nemotron_ds = load_dataset(
        C.NEMOTRON_DATASET, C.NEMOTRON_SUBSET,
        split=C.NEMOTRON_SPLIT, streaming=True, token=token
    ).take(C.NEMOTRON_EVAL_ROWS)
    
    for row in nemotron_ds:
        text = normalize_text(row.get(C.NEMOTRON_TEXT_COL, "") or "")
        if text: eval_data.append(text)

    deepmath_ds = load_dataset(
        C.DEEPMATH_DATASET, split=C.DEEPMATH_SPLIT,
        streaming=True, token=token
    ).take(C.DEEPMATH_EVAL_ROWS)

    for row in deepmath_ds:
        text = normalize_text(build_deepmath_text(row))
        if text: eval_data.append(text)
        
    return eval_data

# ─────────────────────────────────────────────────────────────
# 4. Tokenizer Engineering
# ─────────────────────────────────────────────────────────────

def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=C.UNK_TOKEN))
    tokenizer.normalizer = NormSequence([
        NFC(), Strip(),
        Replace("\u00a0", " "), Replace("\u2009", " "),
        Replace("\u202f", " "), Replace("\u2003", " "),
        Replace("\u2002", " "), Replace("\u200b", ""), Replace("\ufeff", ""),
    ])
    tokenizer.pre_tokenizer = Split(pattern=C.PRETOKENIZER_REGEX, behavior="isolated")
    return tokenizer

def train_tokenizer(tokenizer: Tokenizer, iterator: TrainingCorpusIterator) -> Tokenizer:
    # Hard cap worker threads slightly below max vCPUs to give the process breathing room
    os.environ["RAYON_NUM_THREADS"] = "2"
    log.info(f"Enforcing safe fallback threading limits: RAYON_NUM_THREADS = {os.environ['RAYON_NUM_THREADS']}")

    trainer = BpeTrainer(
        vocab_size=C.VOCAB_SIZE,
        min_frequency=C.MIN_FREQUENCY,
        special_tokens=C.SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=[],
        continuing_subword_prefix="##",
        end_of_word_suffix="",
    )
    log.info("Executing inline Rust BPE engine via bounded batch queues...")
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    return tokenizer

def wrap_as_fast_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=C.UNK_TOKEN, bos_token=C.BOS_TOKEN,
        eos_token=C.EOS_TOKEN, pad_token=C.PAD_TOKEN,
        model_max_length=4096, padding_side="right", truncation_side="right",
    )
    fast.add_special_tokens({
        "additional_special_tokens": [
            t for t in C.SPECIAL_TOKENS 
            if t not in (C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN, C.PAD_TOKEN)
        ]
    })
    return fast

# ─────────────────────────────────────────────────────────────
# 5. Metrics & Validation Suite
# ─────────────────────────────────────────────────────────────

def _encode(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        res = tokenizer.encode(text)
        return res if isinstance(res, list) else res.ids
    return tokenizer(text)["input_ids"]

def calculate_metrics(tokenizer, name: str, eval_samples: List[str]) -> Tuple[Dict, Dict]:
    log.info(f"  Evaluating metrics for [{name}]...")
    
    total_words = total_tokens = unk_count = continued = 0
    math_ops_seen, math_ops_as_single = set(), set()
    
    unk_id = None
    try:
        unk_id = (tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}).get(C.UNK_TOKEN)
    except Exception:
        pass

    for line in eval_samples:
        words = line.split()
        ids = _encode(tokenizer, line)
        total_words += len(words)
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

    # Corpus Summary Data
    fertility = total_tokens / max(total_words, 1)
    pcw = continued / max(total_tokens, 1)
    unk_rate = unk_count / max(total_tokens, 1)
    math_cov = len(math_ops_as_single) / max(len(math_ops_seen), 1) if math_ops_seen else 0.0

    corpus_res = {
        "fertility": round(fertility, 4),
        "continued_word_pct": round(pcw * 100, 2),
        "unk_rate": round(unk_rate * 100, 4),
        "math_symbol_coverage": round(math_cov * 100, 2),
        "vocab_size": tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else len(tokenizer.get_vocab()),
    }

    # Stress Test Suitability Deterministic Check
    digit_pass = sum(1 for d in C.PROBE_DIGITS if len(_encode(tokenizer, d)) == 1 and (unk_id is None or _encode(tokenizer, d)[0] != unk_id))
    digit_integrity = digit_pass / len(C.PROBE_DIGITS)

    tok_counts = [len(_encode(tokenizer, n)) for n in C.PROBE_NUMBERS]
    multi = [n for n in C.PROBE_NUMBERS if sum(c.isdigit() for c in n) > 1]
    pcw_numbers = sum(1 for n in multi if len(_encode(tokenizer, n)) > 1) / max(len(multi), 1)

    strr_pass = sum(1 for cmd in C.PROBE_LATEX_COMMANDS if len(_encode(tokenizer, cmd)) == 1 and (unk_id is None or _encode(tokenizer, cmd)[0] != unk_id))
    
    total_chars = sum(len(e.replace(" ", "")) for e in C.PROBE_EXPRESSIONS)
    total_etoks = sum(len(_encode(tokenizer, e)) for e in C.PROBE_EXPRESSIONS)

    stress_res = {
        "digit_integrity_rate": round(digit_integrity * 100, 2),
        "arithmetic_readiness_mean_toks": round(float(np.mean(tok_counts)), 3),
        "strr_latex_pct": round((strr_pass / len(C.PROBE_LATEX_COMMANDS)) * 100, 2),
        "pcw_numbers_fragmented_pct": round(pcw_numbers * 100, 2),
        "cpt_expressions": round(total_chars / max(total_etoks, 1), 3),
    }

    return corpus_res, stress_res

def print_results_table(corpus_results: Dict, stress_results: Dict) -> None:
    # Keeps verification output standard
    for t, title, cols in [("corpus", "A. Corpus Metrics (Held-out Set)", [("fertility", "Fertility↓"), ("continued_word_pct", "Cont.Word%↓"), ("unk_rate", "UNK Rate↓"), ("math_symbol_coverage", "MathSym%↑"), ("vocab_size", "Vocab")]),
                           ("stress", "B. Stress-Test Suite", [("digit_integrity_rate", "Digit Integrity%↑"), ("arithmetic_readiness_mean_toks", "ArithReady(toks)↓"), ("strr_latex_pct", "STRR LaTeX%↑"), ("pcw_numbers_fragmented_pct", "NumFrag%↑"), ("cpt_expressions", "CPT Expr↑")])]:
        print(f"\n{'═'*90}\n  {title}\n{'═'*90}")
        print(f"{'Model':<26}" + "".join(f"{h:<22}" for _, h in cols) + f"\n{'─'*90}")
        res_dict = corpus_results if t == "corpus" else stress_results
        for name, row in res_dict.items():
            print(f"{name:<26}" + "".join(f"{f'{row.get(k):,}' if isinstance(row.get(k), int) else f'{row.get(k):.3f}':<22}" for k, _ in cols))

# ─────────────────────────────────────────────────────────────
# 6. Main Sequence Execution
# ─────────────────────────────────────────────────────────────

def main() -> None:
    wall_start = time.time()
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"
    
    _log_memory("Initialization")
    eval_samples = get_eval_set(HF_TOKEN)

    # Instantiate Direct streaming iterator configuration
    corpus_iterator = TrainingCorpusIterator(token=HF_TOKEN, word_cap=C.TRAIN_WORD_CAP)

    with timer("BPE Native Tokenizer Training"):
        tokenizer = build_tokenizer()
        tokenizer = train_tokenizer(tokenizer, corpus_iterator)

    _log_memory("Post-Training Execution")

    # Fast Transformation Export Phase
    tokenizer.save(str(TOK_JSON))
    fast_tok = wrap_as_fast_tokenizer(tokenizer)
    fast_tok.save_pretrained(str(TOK_DIR))

    corpus_res, stress_res = {}, {}
    
    with timer("Evaluating MathFormer Pipeline"):
        c_res, s_res = calculate_metrics(tokenizer, "MathFormer", eval_samples)
        corpus_res["MathFormer"] = c_res
        stress_res["MathFormer"] = s_res

    # Unload primary variables to handle isolated validation allocations safely
    del tokenizer
    gc.collect()

    # Load and clean benchmark allocations inside process blocks safely
    from transformers import AutoTokenizer
    for display_name, hf_id in C.REFERENCE_TOKENIZERS:
        with timer(f"Evaluating Baseline Reference: {display_name}"):
            try:
                ref_tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN, trust_remote_code=True)
                backend = getattr(ref_tok, "_tokenizer", None) or ref_tok
                c_res, s_res = calculate_metrics(backend, display_name, eval_samples)
                corpus_res[display_name] = c_res
                stress_res[display_name] = s_res
                del ref_tok, backend
            except Exception as e:
                log.warning(f"Failed to cleanly track metadata baseline target metrics for {display_name}: {e}")
            finally:
                gc.collect()

    print_results_table(corpus_res, stress_res)

    # Sanity Assert Gates
    if corpus_res.get("MathFormer", {}).get("unk_rate", 99.0) > 0.5:
        log.warning("❗ Out-of-vocabulary fallback tracking rate spikes above target thresholds.")
    if stress_res.get("MathFormer", {}).get("digit_integrity_rate", 100.0) < 100.0:
        log.warning("❗ Mutation tracking anomaly caught: Numeric characters are fracturing on processing boundaries.")

    with timer("HuggingFace Hub Target Synchronisation"):
        log.info(f"Pushing payload assets to: {C.HUB_REPO_ID}")
        fast_tok.push_to_hub(C.HUB_REPO_ID, commit_message=C.HUB_COMMIT_MESSAGE, token=HF_TOKEN, private=False)

    log.info(f"🏁 Complete execution window closed in: {timedelta(seconds=int(time.time() - wall_start))}")

if __name__ == "__main__":
    main()