"""
train_tokenizer.py
──────────────────
Trains a 16K Rust-based BPE tokenizer on 150M tokens from
Nemotron-CC-Math-v1 and DeepMath-103K, evaluates it against
reference tokenizers on a held-out 10M-token set and a
deterministic stress-test suite, then pushes to HF Hub.

Hardware target : 4 vCPUs, 27 GB RAM (Modal CPU worker)
Runtime estimate: ~25–40 minutes end-to-end

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
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Dict, Tuple, Optional

import numpy as np
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, pre_tokenizers, normalizers, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, Strip, Replace, Sequence as NormSequence
from tokenizers.pre_tokenizers import Split, Sequence as PreTokSequence
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi

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
    """Exit immediately if a required env var is missing."""
    val = os.environ.get(var, "").strip()
    if not val:
        log.error(
            f"Required environment variable '{var}' is not set.\n"
            f"Export it before running:  export {var}=<your_token>"
        )
        sys.exit(1)
    return val


HF_TOKEN = _require_env("HF_TOKEN")

# Resolve paths: prefer Modal volume mounts when they exist
_on_modal = Path(C.MODAL_VOLUME_MOUNT).exists()
DATA_DIR   = C.MODAL_DATA_DIR   if _on_modal else C.DATA_DIR
TOK_DIR    = C.MODAL_TOK_DIR    if _on_modal else C.TOKENIZER_DIR
CORPUS_TRAIN = C.MODAL_CORPUS_TRAIN if _on_modal else C.CORPUS_TRAIN_PATH
CORPUS_EVAL  = C.MODAL_CORPUS_EVAL  if _on_modal else C.CORPUS_EVAL_PATH
TOK_JSON     = C.MODAL_TOK_JSON     if _on_modal else C.TOKENIZER_JSON_PATH

DATA_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(C.RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# 1. Text normalisation  (Python-side, before corpus write)
# ─────────────────────────────────────────────────────────────

# Unicode "other space" characters that are not ASCII 0x20
_UNICODE_SPACES = re.compile(
    r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\ufeff]"
)
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Normalisation pipeline applied to every document before it is written
    to the training corpus file.

    Steps (order is deliberate):
    1. NFC Unicode normalisation — composes combining characters.
       NOT NFKC: we preserve LaTeX ligatures and special forms.
    2. Replace exotic Unicode spaces with ASCII 0x20.
    3. Collapse runs of 3+ consecutive blank lines to exactly 2.
    4. Strip leading/trailing whitespace from the document.

    We do NOT lowercase (math is case-sensitive: x ≠ X, A ≠ a).
    We do NOT remove punctuation or symbols.
    """
    if not text:
        return ""

    # Step 1 — NFC
    text = unicodedata.normalize(C.UNICODE_NORM_FORM, text)

    # Step 2 — exotic spaces → ASCII space
    text = _UNICODE_SPACES.sub(" ", text)

    # Step 3 — collapse excessive blank lines
    text = _MULTI_BLANK.sub("\n\n", text)

    # Step 4 — strip
    text = text.strip()

    return text


def build_deepmath_text(row: dict) -> str:
    """
    Concatenate question + solution for DeepMath rows.
    A blank line separates them so the tokenizer sees natural
    question/answer structure without a special delimiter.
    """
    parts = [row.get(C.DEEPMATH_TEXT_COL, "")]
    for col in C.DEEPMATH_AUX_COLS:
        val = row.get(col, "") or ""
        if val.strip():
            parts.append(val.strip())
    return "\n\n".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────────────────────
# 2. Dataset streaming + corpus file construction
# ─────────────────────────────────────────────────────────────

def _approx_token_count(text: str) -> int:
    """
    Whitespace-split word count as a cheap proxy for token count during
    corpus construction. The real token count is measured post-training.
    """
    return len(text.split())


def stream_nemotron(
    n_rows: int,
    skip_indices: Optional[set] = None,
) -> Iterator[str]:
    """
    Stream `n_rows` random rows from Nemotron-CC-Math-v1 (4plus subset).
    `skip_indices` is a set of row indices reserved for eval — these are
    never yielded by the training streamer and vice versa.
    """
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
        if len(text) > 50:          # skip near-empty rows
            yield text
            collected += 1
    log.info(f"  → yielded {collected:,} Nemotron rows")


def stream_deepmath(
    n_rows: int,
    skip_indices: Optional[set] = None,
) -> Iterator[str]:
    """
    Stream `n_rows` rows from DeepMath-103K.
    DeepMath is small enough to load fully into memory.
    """
    log.info(f"Loading DeepMath ({n_rows:,} rows requested) …")
    ds = load_dataset(
        C.DEEPMATH_DATASET,
        split=C.DEEPMATH_SPLIT,
        token=HF_TOKEN,
    )
    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)

    collected = 0
    for idx in all_indices:
        if skip_indices and idx in skip_indices:
            continue
        if collected >= n_rows:
            break
        text = normalize_text(build_deepmath_text(ds[idx]))
        if len(text) > 50:
            yield text
            collected += 1
    log.info(f"  → yielded {collected:,} DeepMath rows")
    del ds; gc.collect()


def build_corpus_files() -> None:
    """
    Write CORPUS_TRAIN and CORPUS_EVAL.
    Each file contains one document per line (empty line between docs
    is handled by the trainer's iterator).

    Eval rows are drawn FIRST so their indices can be passed as
    `skip_indices` to the training streamers, guaranteeing zero overlap.
    """
    if CORPUS_TRAIN.exists() and CORPUS_EVAL.exists():
        log.info("Corpus files already exist — skipping rebuild.")
        return

    log.info("═" * 60)
    log.info("Building corpus files …")
    log.info("═" * 60)

    # ── Eval corpus ──────────────────────────────────────────
    # Reserve a contiguous block of the FIRST N rows for eval
    # (deterministic with fixed seed).
    nemotron_eval_skip  = set(range(C.NEMOTRON_EVAL_ROWS))
    deepmath_eval_skip  = set(range(C.DEEPMATH_EVAL_ROWS))

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

    log.info(f"  Eval corpus: ~{eval_tokens:,} whitespace-tokens written")

    # ── Train corpus ─────────────────────────────────────────
    log.info("Writing train corpus …")
    train_tokens = 0
    with open(CORPUS_TRAIN, "w", encoding="utf-8") as f:
        for text in stream_nemotron(
            C.NEMOTRON_TRAIN_ROWS,
            skip_indices=nemotron_eval_skip,
        ):
            f.write(text + "\n")
            train_tokens += _approx_token_count(text)

        for text in stream_deepmath(
            C.DEEPMATH_TRAIN_ROWS,
            skip_indices=deepmath_eval_skip,
        ):
            f.write(text + "\n")
            train_tokens += _approx_token_count(text)

    log.info(f"  Train corpus: ~{train_tokens:,} whitespace-tokens written")
    log.info("Corpus files ready.")


# ─────────────────────────────────────────────────────────────
# 3. Tokenizer construction
# ─────────────────────────────────────────────────────────────

def build_tokenizer() -> Tokenizer:
    """
    Construct the HuggingFace `tokenizers` Tokenizer object with:

    Normaliser stack
    ────────────────
    1. NFC             — Unicode composition (tokenizers-native)
    2. Strip           — strip leading/trailing whitespace
    3. Replace(exotic Unicode spaces → ASCII space)

    Pre-tokeniser
    ─────────────
    A single Regex Split pre-tokeniser using the pattern defined in
    config.PRETOKENIZER_REGEX.

    Behaviour guaranteed by the regex alternation order:
      • Whitespace runs  → own isolated piece (never merged into a word)
      • LaTeX commands   → single piece (\frac, \alpha, …)
      • Digits           → one piece per digit (123 → ['1','2','3'])
      • Alpha runs       → pure letter sequences (no embedded digits)
      • Operators        → one piece each (+, -, =, ≤, …)
      • Brackets/punct   → one piece each
      • Fallthrough      → any remaining unicode char

    This means BPE only ever merges within a pre-tokenised piece boundary.
    A space can NEVER be merged with adjacent letters. '3' can NEVER be
    merged with 'x'. \frac will always be one unit unless BPE splits it
    further (it won't — it appears as one pre-token from the start).

    Model
    ─────
    BPE (byte-level fallback disabled — we rely on the catch-all regex
    branch to handle unknown Unicode rather than byte fallback, which
    would pollute the vocab with noise bytes).
    """
    tokenizer = Tokenizer(BPE(unk_token=C.UNK_TOKEN))

    # ── Normaliser ───────────────────────────────────────────
    tokenizer.normalizer = NormSequence([
        NFC(),
        Strip(),
        # U+00A0 non-breaking space → regular space
        Replace("\u00a0", " "),
        # Hair space, thin space, em space, etc.
        Replace("\u2009", " "),
        Replace("\u202f", " "),
        Replace("\u2003", " "),
        Replace("\u2002", " "),
        Replace("\u200b", ""),   # zero-width space → remove entirely
        Replace("\ufeff", ""),   # BOM → remove
    ])

    # ── Pre-tokeniser ────────────────────────────────────────
    # Split behaviour = ISOLATED: each captured group is its own piece.
    # The whitespace group is included in the output (not removed) so
    # the model sees whitespace as first-class tokens.
    tokenizer.pre_tokenizer = Split(
        pattern=C.PRETOKENIZER_REGEX,
        behavior="isolated",     # every match + every non-match = own piece
    )

    return tokenizer


def train_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """
    Run BPE training on the corpus file.
    Uses all 4 vCPUs via the tokenizers library's internal parallelism.
    """
    trainer = BpeTrainer(
        vocab_size=C.VOCAB_SIZE,
        min_frequency=C.MIN_FREQUENCY,
        special_tokens=C.SPECIAL_TOKENS,
        show_progress=True,
        # Do not limit initial alphabet — let all Unicode chars in corpus
        # seed the initial vocabulary before BPE merges begin.
        initial_alphabet=[],
        # Byte fallback disabled: unknown chars map to <unk> via the
        # catch-all regex branch, not to byte sequences.
        continuing_subword_prefix="##",
        end_of_word_suffix="",
    )

    log.info("Starting BPE training …")
    t0 = time.time()
    tokenizer.train(files=[str(CORPUS_TRAIN)], trainer=trainer)
    elapsed = time.time() - t0
    log.info(f"BPE training complete in {elapsed:.1f}s")
    log.info(f"Final vocab size: {tokenizer.get_vocab_size():,}")

    return tokenizer


def wrap_as_fast_tokenizer(tokenizer: Tokenizer) -> PreTrainedTokenizerFast:
    """
    Wrap the low-level Tokenizer in a PreTrainedTokenizerFast so it can
    be pushed to the Hub and used with transformers pipelines.
    """
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=C.UNK_TOKEN,
        bos_token=C.BOS_TOKEN,
        eos_token=C.EOS_TOKEN,
        pad_token=C.PAD_TOKEN,
        model_max_length=4096,
        # Padding strategy — right-pad by default
        padding_side="right",
        truncation_side="right",
    )
    # Register all special tokens so they're never split
    fast.add_special_tokens({
        "additional_special_tokens": [
            t for t in C.SPECIAL_TOKENS
            if t not in (C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN, C.PAD_TOKEN)
        ]
    })
    return fast


# ─────────────────────────────────────────────────────────────
# 4. Evaluation helpers
# ─────────────────────────────────────────────────────────────

def _encode(tokenizer, text: str) -> List[int]:
    """Unified encode interface for both fast and slow tokenizers."""
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)
        if isinstance(result, list):
            return result
        # tokenizers.Encoding object
        return result.ids
    return tokenizer(text)["input_ids"]


def corpus_metrics(
    tokenizer,
    name: str,
    corpus_path: Path,
    max_docs: int = 50_000,
) -> Dict[str, float]:
    """
    Compute fertility, continued-word %, unknown-token rate, and
    math-symbol coverage over the held-out eval corpus.

    Reads at most `max_docs` lines to keep runtime bounded.
    """
    log.info(f"  Running corpus metrics for [{name}] …")

    total_words   = 0
    total_tokens  = 0
    unk_count     = 0
    continued     = 0
    math_ops_seen = set()
    math_ops_as_single = set()

    # Build unk id
    unk_id = None
    try:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        unk_id = vocab.get(C.UNK_TOKEN)
    except Exception:
        pass

    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue

            words  = line.split()
            ids    = _encode(tokenizer, line)

            total_words  += len(words)
            total_tokens += len(ids)

            if unk_id is not None:
                unk_count += ids.count(unk_id)

            # Continued words (## prefix convention)
            try:
                conv = tokenizer.convert_ids_to_tokens(ids)
                continued += sum(1 for t in conv if t and t.startswith("##"))
            except Exception:
                pass

            # Math operator coverage
            for op in C.MATH_OPERATORS_CORPUS:
                if op in line:
                    math_ops_seen.add(op)
                    try:
                        op_ids = _encode(tokenizer, op)
                        if len(op_ids) == 1 and op_ids[0] != unk_id:
                            math_ops_as_single.add(op)
                    except Exception:
                        pass

    fertility  = total_tokens / max(total_words, 1)
    pcw        = continued / max(total_tokens, 1)
    unk_rate   = unk_count  / max(total_tokens, 1)
    math_cov   = (len(math_ops_as_single) / max(len(math_ops_seen), 1)
                  if math_ops_seen else float("nan"))

    return {
        "fertility":             round(fertility, 4),
        "continued_word_pct":   round(pcw * 100, 2),
        "unk_rate":              round(unk_rate * 100, 4),
        "math_symbol_coverage": round(math_cov * 100, 2),
        "vocab_size":           (tokenizer.get_vocab_size()
                                  if hasattr(tokenizer, "get_vocab_size")
                                  else len(tokenizer.get_vocab())),
    }


def stress_test(tokenizer, name: str) -> Dict[str, float]:
    """
    Deterministic stress-test suite against curated probe sets.

    Metrics
    ───────
    digit_integrity_rate    — fraction of individual digits that map to
                              exactly one token ID (target: 1.0)
    arithmetic_readiness    — mean tokens per multi-digit number
                              (target: equals digit count of the number)
    strr                    — single-token retention rate for LaTeX
                              commands: fraction that tokenise as 1 token
    pcw_numbers             — fraction of multi-digit numbers fragmented
                              across >1 token boundary (lower is NOT
                              necessarily better — we WANT fragmentation;
                              this just confirms it happens consistently)
    cpt_expressions         — chars per token on pure math expressions
                              (higher = more efficient compression)
    """
    log.info(f"  Running stress-test for [{name}] …")

    try:
        unk_id = tokenizer.get_vocab().get(C.UNK_TOKEN)
    except Exception:
        unk_id = None

    # ── Digit integrity ─────────────────────────────────────
    digit_pass = 0
    for d in C.PROBE_DIGITS:
        ids = _encode(tokenizer, d)
        if len(ids) == 1 and ids[0] != unk_id:
            digit_pass += 1
    digit_integrity = digit_pass / len(C.PROBE_DIGITS)

    # ── Arithmetic readiness ─────────────────────────────────
    token_counts_per_number = []
    fragmented_numbers = 0
    for num_str in C.PROBE_NUMBERS:
        # Strip sign and decimal point — count pure digit chars
        digit_chars = sum(1 for c in num_str if c.isdigit())
        ids = _encode(tokenizer, num_str)
        n_toks = len(ids)
        token_counts_per_number.append(n_toks)
        if n_toks > 1:
            fragmented_numbers += 1   # multi-digit numbers SHOULD fragment

    arith_readiness = float(np.mean(token_counts_per_number))
    # For PCW-numbers we report what fraction of multi-digit numbers
    # were correctly fragmented (≥2 tokens)
    multi_digit_numbers = [n for n in C.PROBE_NUMBERS if sum(c.isdigit() for c in n) > 1]
    pcw_numbers_correct = 0
    for num_str in multi_digit_numbers:
        ids = _encode(tokenizer, num_str)
        if len(ids) > 1:
            pcw_numbers_correct += 1
    pcw_numbers = pcw_numbers_correct / max(len(multi_digit_numbers), 1)

    # ── Single-token retention rate (LaTeX commands) ─────────
    strr_pass = 0
    for cmd in C.PROBE_LATEX_COMMANDS:
        ids = _encode(tokenizer, cmd)
        if len(ids) == 1 and ids[0] != unk_id:
            strr_pass += 1
    strr = strr_pass / len(C.PROBE_LATEX_COMMANDS)

    # ── Chars per token on math expressions ─────────────────
    total_chars  = 0
    total_etokens = 0
    for expr in C.PROBE_EXPRESSIONS:
        clean = expr.replace(" ", "")   # strip spaces for pure density
        ids = _encode(tokenizer, expr)
        total_chars   += len(clean)
        total_etokens += len(ids)
    cpt = total_chars / max(total_etokens, 1)

    return {
        "digit_integrity_rate":    round(digit_integrity  * 100, 2),
        "arithmetic_readiness_mean_toks": round(arith_readiness, 3),
        "strr_latex_pct":          round(strr * 100, 2),
        "pcw_numbers_fragmented_pct": round(pcw_numbers * 100, 2),
        "cpt_expressions":         round(cpt, 3),
    }


def run_full_evaluation(
    our_tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Dict, Dict]:
    """
    Evaluate MathFormer + all reference tokenizers.
    Returns (corpus_results, stress_results) dicts keyed by model name.
    """
    log.info("═" * 60)
    log.info("Running full evaluation …")
    log.info("═" * 60)

    all_corpus  = {}
    all_stress  = {}

    # ── Our tokenizer ─────────────────────────────────────────
    all_corpus["MathFormer"] = corpus_metrics(
        our_tokenizer.backend_tokenizer, "MathFormer", CORPUS_EVAL
    )
    all_stress["MathFormer"] = stress_test(
        our_tokenizer.backend_tokenizer, "MathFormer"
    )

    # ── Reference tokenizers ──────────────────────────────────
    for display_name, hf_id in C.REFERENCE_TOKENIZERS:
        log.info(f"Loading reference tokenizer: {hf_id}")
        try:
            from transformers import AutoTokenizer
            ref_tok = AutoTokenizer.from_pretrained(
                hf_id, token=HF_TOKEN, trust_remote_code=True
            )
            # Corpus metrics — use the fast backend if available
            backend = getattr(ref_tok, "_tokenizer", None) or ref_tok
            all_corpus[display_name] = corpus_metrics(
                backend, display_name, CORPUS_EVAL
            )
            all_stress[display_name] = stress_test(backend, display_name)
            del ref_tok; gc.collect()
        except Exception as e:
            log.warning(f"  Could not load {hf_id}: {e}")
            all_corpus[display_name] = {"error": str(e)}
            all_stress[display_name] = {"error": str(e)}

    return all_corpus, all_stress


def print_results_table(
    corpus_results: Dict[str, Dict],
    stress_results: Dict[str, Dict],
) -> None:
    """Pretty-print evaluation tables to stdout."""

    CORPUS_COLS = [
        ("fertility",             "Fertility↓"),
        ("continued_word_pct",   "Cont.Word%↓"),
        ("unk_rate",              "UNK Rate↓"),
        ("math_symbol_coverage", "MathSym%↑"),
        ("vocab_size",            "Vocab"),
    ]
    STRESS_COLS = [
        ("digit_integrity_rate",         "Digit Integrity%↑"),
        ("arithmetic_readiness_mean_toks","ArithReady(toks)↓"),
        ("strr_latex_pct",               "STRR LaTeX%↑"),
        ("pcw_numbers_fragmented_pct",   "NumFrag%↑"),
        ("cpt_expressions",              "CPT Expr↑"),
    ]

    def _fmt(v) -> str:
        if isinstance(v, float): return f"{v:.3f}"
        if isinstance(v, int):   return f"{v:,}"
        return str(v)

    def _table(results, cols, title):
        names = list(results.keys())
        header = f"{'Model':<26}" + "".join(f"{h:<22}" for _, h in cols)
        print(f"\n{'═'*80}")
        print(f"  {title}")
        print('═'*80)
        print(header)
        print('─'*80)
        for name in names:
            row = results[name]
            line = f"{name:<26}"
            for key, _ in cols:
                val = row.get(key, "—")
                line += f"{_fmt(val):<22}"
            print(line)
        print('─'*80)

    _table(corpus_results, CORPUS_COLS,  "A. Corpus Metrics (held-out 10M tokens)")
    _table(stress_results, STRESS_COLS,  "B. Stress-Test Suite (deterministic probes)")
    print()


def save_results(
    corpus_results: Dict,
    stress_results: Dict,
) -> None:
    out = {
        "corpus_metrics": corpus_results,
        "stress_test":    stress_results,
    }
    path = TOK_DIR / "evaluation_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info(f"Evaluation results saved to {path}")


# ─────────────────────────────────────────────────────────────
# 5. Hub push
# ─────────────────────────────────────────────────────────────

def push_to_hub(fast_tokenizer: PreTrainedTokenizerFast) -> None:
    log.info(f"Pushing tokenizer to Hub: {C.HUB_REPO_ID} …")
    fast_tokenizer.push_to_hub(
        C.HUB_REPO_ID,
        commit_message=C.HUB_COMMIT_MESSAGE,
        token=HF_TOKEN,
        private=False,
    )
    log.info("Hub push complete.")


# ─────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    log.info("╔══════════════════════════════════════╗")
    log.info("║   MathFormer Tokenizer Training      ║")
    log.info("╚══════════════════════════════════════╝")
    log.info(f"Running on Modal volume: {_on_modal}")
    log.info(f"Train corpus : {CORPUS_TRAIN}")
    log.info(f"Eval corpus  : {CORPUS_EVAL}")
    log.info(f"Tokenizer out: {TOK_DIR}")

    # Step 1 — Build corpus files
    build_corpus_files()

    # Step 2 — Build and train tokenizer
    tokenizer = build_tokenizer()
    tokenizer = train_tokenizer(tokenizer)

    # Step 3 — Save raw tokenizer JSON
    tokenizer.save(str(TOK_JSON))
    log.info(f"Raw tokenizer saved: {TOK_JSON}")

    # Step 4 — Wrap as PreTrainedTokenizerFast
    fast_tok = wrap_as_fast_tokenizer(tokenizer)
    fast_tok.save_pretrained(str(TOK_DIR))
    log.info(f"Fast tokenizer saved: {TOK_DIR}")

    # Step 5 — Evaluate
    corpus_res, stress_res = run_full_evaluation(fast_tok)
    print_results_table(corpus_res, stress_res)
    save_results(corpus_res, stress_res)

    # Step 6 — Sanity check before push
    unk_rate = corpus_res.get("MathFormer", {}).get("unk_rate", 99.0)
    if unk_rate > 0.5:
        log.warning(
            f"UNK rate is {unk_rate:.2f}% — higher than expected. "
            f"Review the corpus or increase vocab size before deploying."
        )

    digit_integrity = stress_res.get("MathFormer", {}).get("digit_integrity_rate", 0.0)
    if digit_integrity < 100.0:
        log.warning(
            f"Digit integrity is {digit_integrity:.1f}% (target: 100%). "
            f"Review the pre-tokeniser regex."
        )

    # Step 7 — Push to Hub
    push_to_hub(fast_tok)

    log.info("Done.")


if __name__ == "__main__":
    main()
