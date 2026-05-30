"""
train_sentencepiece_native.py
─────────────────────────────
Streams, normalizes, and balances datasets to a local flat file,
then executes a native C++ SentencePiece Unigram training iteration.
Converts the final asset to a Transformers Fast Tokenizer for Hugging Face deployment.
"""

import itertools
import logging
import os
import re
import sys
import time
import unicodedata
from datetime import timedelta
from pathlib import Path

from tqdm import tqdm

import sentencepiece as spm
from datasets import load_dataset
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace as MetaspacePreTokenizer
from transformers import PreTrainedTokenizerFast

import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mathformer_spm")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    log.error("Missing HF_TOKEN env variable.")
    sys.exit(1)

OUTPUT_DIR = Path("./mathformer_16k_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_CORPUS_FILE = OUTPUT_DIR / "normalized_nemotron_corpus.txt"
SPM_MODEL_PREFIX = str(OUTPUT_DIR / "spm_core_model")

# ─────────────────────────────────────────────────────────────
# 1. Normalization Flow
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

# ─────────────────────────────────────────────────────────────
# 2. Stage 1: Write Normalized Raw Text to Disk
# ─────────────────────────────────────────────────────────────
def build_deepmath_text(row: dict) -> str:
    """
    Extracts and maps a DeepMath row into a structured sequence string.
    Ensures question, reasoning steps, and final answers are bound together.
    """
    question = row.get("question", "") or ""
    thought = row.get("r1_solution_1", "") or ""
    output = row.get("final_answer", "") or ""
    return f"Question: {question}\nThought: {thought}\nFinal Answer: {output}"

def build_local_corpus():
    if TMP_CORPUS_FILE.exists():
        log.info(f"💾 Found existing normalized corpus file at {TMP_CORPUS_FILE}. Skipping extraction.")
        return

    log.info(f"📝 Creating a balanced, normalized corpus file: {TMP_CORPUS_FILE}")
    word_count = 0

    target_per_dataset = C.TRAIN_WORD_CAP // 2
    pbar = tqdm(total=C.TRAIN_WORD_CAP, desc="✍️ Writing balanced lines to disk", unit="word")

    with open(TMP_CORPUS_FILE, "w", encoding="utf-8") as f:
        # ─── STREAM 1: NEMOTRON ───
        log.info(f"🧬 Extracting up to {target_per_dataset:,} words from Nemotron...")
        nemotron_words = 0
        nemotron_ds = load_dataset(
            C.NEMOTRON_DATASET, C.NEMOTRON_SUBSET,
            split=C.NEMOTRON_SPLIT, streaming=True, token=HF_TOKEN
        ).skip(C.NEMOTRON_EVAL_ROWS)

        # islice caps iteration without breaking mid-stream, avoiding the
        # PyGILState_Release crash caused by abandoning background fetch threads.
        for row in itertools.islice(nemotron_ds, 500_000):
            if nemotron_words >= target_per_dataset:
                continue
            text = normalize_text(row.get(C.NEMOTRON_TEXT_COL, "") or "")
            if len(text) > 50:
                words_in_row = len(text.split())
                nemotron_words += words_in_row
                word_count += words_in_row
                cleaned_line = text.replace("\n", " ")
                f.write(cleaned_line + "\n")
                pbar.update(words_in_row)

        # ─── STREAM 2: DEEPMATH ───
        log.info(f"🧬 Extracting up to {target_per_dataset:,} words from DeepMath...")
        deepmath_words = 0
        deepmath_ds = load_dataset(
            C.DEEPMATH_DATASET, split=C.DEEPMATH_SPLIT,
            streaming=True, token=HF_TOKEN
        ).skip(C.DEEPMATH_EVAL_ROWS).shuffle(seed=C.RANDOM_SEED, buffer_size=2_000)

        for row in itertools.islice(deepmath_ds, 500_000):
            if deepmath_words >= target_per_dataset or word_count >= C.TRAIN_WORD_CAP:
                continue
            text = normalize_text(build_deepmath_text(row))
            if len(text) > 50:
                words_in_row = len(text.split())
                deepmath_words += words_in_row
                word_count += words_in_row
                cleaned_line = text.replace("\n", " ")
                f.write(cleaned_line + "\n")
                pbar.update(words_in_row)

    pbar.close()
    log.info(f"✓ Local mixed cache ready. Total balanced words: {word_count:,}")

# ─────────────────────────────────────────────────────────────
# 3. Validation Gate — hard abort before any push
# ─────────────────────────────────────────────────────────────
VALIDATION_CASES = [
    (
        "<think> Let the sequence be x_n = 42 + \\alpha \\cdot \\sum_{i=1}^n \\frac{1}{i^2}. "
        "Therefore, 42 is an upper bound. </think> <answer> The limit converges to 42. </answer>"
    ),
    "<think> base case: n = 1. </think> <answer> Q.E.D. </answer>",
    "<think> step 1. </think> <answer> done. </answer>",
    "<think> </think> <answer> </answer>",
    "No tags, plain math: \\frac{1}{2} + \\sum_{i=1}^{n} i^2 = 42.",
    "<think> x = 123 + 456. </think> <answer> 579. </answer>",
]

def validate_roundtrip(tokenizer: PreTrainedTokenizerFast) -> None:
    """Hard abort if any test case fails lossless round-trip."""
    failures = []
    for text in VALIDATION_CASES:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        if decoded != text:
            min_len = min(len(text), len(decoded))
            diff_pos = next(
                (i for i in range(min_len) if text[i] != decoded[i]),
                min_len
            )
            failures.append({
                "input":    text,
                "decoded":  decoded,
                "diff_at":  diff_pos,
                "char_in":  repr(text[diff_pos])    if diff_pos < len(text)    else "<missing>",
                "char_out": repr(decoded[diff_pos])  if diff_pos < len(decoded) else "<missing>",
            })

    if failures:
        for f in failures:
            log.error(
                f"Round-trip FAILED:\n"
                f"  IN : {f['input']}\n"
                f"  OUT: {f['decoded']}\n"
                f"  First diff at position {f['diff_at']}: "
                f"expected {f['char_in']}, got {f['char_out']}"
            )
        raise RuntimeError(
            f"Tokenizer validation failed on {len(failures)}/{len(VALIDATION_CASES)} cases. "
            f"Aborting — nothing pushed to Hub."
        )

    log.info(f"✓ All {len(VALIDATION_CASES)} round-trip validation cases passed.")

# ─────────────────────────────────────────────────────────────
# 4. Build HF Fast Tokenizer directly from SPM vocab
#    (bypasses SpmConverter, which hard-codes prepend_scheme="first"
#     in its Rust layer and silently ignores post-hoc decoder assignment)
# ─────────────────────────────────────────────────────────────
def build_fast_tokenizer(spm_model_path: str, special_tokens: list[str]) -> Tokenizer:
    """
    Reads the SPM vocab directly and constructs a tokenizers.Tokenizer with
    a Metaspace decoder locked to prepend_scheme='always'.  This is the only
    way to guarantee the decoder setting is respected in older tokenizers builds
    where SpmConverter's internal Rust construction overwrites any Python-side
    decoder assignment.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)

    # Extract full vocab as (piece, score) pairs — preserves exact SPM ordering
    vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]

    # Build the Unigram model directly
    tokenizer = Tokenizer(Unigram(vocab, unk_id=0))

    # Pre-tokenizer: split on whitespace and prepend ▁ — matches SPM behaviour
    tokenizer.pre_tokenizer = MetaspacePreTokenizer(
        replacement="\u2581",       # ▁  (U+2581)
        prepend_scheme="always",    # every token at a word boundary gets ▁
        split=True,
    )

    # Decoder: inverse of pre-tokenizer — prepend_scheme='always' is what
    # restores spaces before closing tags and between adjacent special tokens.
    tokenizer.decoder = MetaspaceDecoder(
        replacement="\u2581",
        prepend_scheme="always",
    )

    # Register all special tokens so they are never split by the pre-tokenizer
    added = [
        AddedToken(tok, single_word=False, lstrip=False, rstrip=False, normalized=False)
        for tok in special_tokens
    ]
    tokenizer.add_special_tokens(added)

    return tokenizer

# ─────────────────────────────────────────────────────────────
# 5. Stage 2: Train Native SentencePiece & Convert
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    build_local_corpus()

    log.info("⚙️ Compiling custom user-defined structural and mathematical tokens...")

    base_specials = {C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN, C.PAD_TOKEN}
    user_symbols = [t for t in C.SPECIAL_TOKENS if t not in base_specials]

    if hasattr(C, "ADDITIONAL_MATH_SYMBOLS"):
        user_symbols.extend(C.ADDITIONAL_MATH_SYMBOLS)

    # FIX B: register ▁-prefixed closing-tag variants so SPM encodes
    # "▁</think>" as one atomic piece rather than fragmenting the space.
    closing_tag_variants = [
        f"\u2581{tok}"
        for tok in user_symbols
        if tok.startswith("</")
    ]
    user_symbols = list(dict.fromkeys(user_symbols + closing_tag_variants))

    log.info("🔨 Launching Native C++ SentencePiece Unigram Trainer...")

    spm.SentencePieceTrainer.train(
        input=str(TMP_CORPUS_FILE),
        model_prefix=SPM_MODEL_PREFIX,
        vocab_size=C.VOCAB_SIZE,
        model_type="unigram",
        normalization_rule_name="nfkc",
        max_sentencepiece_length=16,
        split_digits=True,
        user_defined_symbols=",".join(user_symbols),
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece=C.UNK_TOKEN,
        bos_piece=C.BOS_TOKEN,
        eos_piece=C.EOS_TOKEN,
        pad_piece=C.PAD_TOKEN,
        character_coverage=1.0,
        max_sentence_length=100000,
        hard_vocab_limit=False,
        byte_fallback=False,
        split_by_whitespace=False,
        remove_extra_whitespaces=False,
    )
    log.info("✓ Native SentencePiece training cycle complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # FIX C (proper): build tokenizer directly — skip SpmConverter entirely.
    # SpmConverter's Rust construction hard-codes prepend_scheme="first" at the
    # C++ layer; any Python-side .decoder = ... assignment is silently dropped.
    # ─────────────────────────────────────────────────────────────────────────
    log.info("⚡ Building Fast Tokenizer directly from SPM vocab (bypassing SpmConverter)...")

    all_special_tokens = list(base_specials) + user_symbols
    fast_tokenizer_object = build_fast_tokenizer(
        spm_model_path=f"{SPM_MODEL_PREFIX}.model",
        special_tokens=all_special_tokens,
    )

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=fast_tokenizer_object,
        unk_token=C.UNK_TOKEN,
        bos_token=C.BOS_TOKEN,
        eos_token=C.EOS_TOKEN,
        pad_token=C.PAD_TOKEN,
        model_max_length=4096,
        padding_side="right",
        add_prefix_space=True,
    )

    # Hard abort before save or push if any round-trip fails.
    log.info("🧪 Running round-trip validation gate...")
    validate_roundtrip(fast_tok)

    fast_tok.save_pretrained(str(OUTPUT_DIR))
    log.info(f"💾 Converted fast configuration assets saved to: {OUTPUT_DIR}")

    log.info(f"🚀 Pushing final assets to Hub: {C.HUB_REPO_ID}")
    fast_tok.push_to_hub(
        C.HUB_REPO_ID,
        commit_message="Deploy native SentencePiece-backed 16K Unigram tokenizer with verified space preservation.",
        token=HF_TOKEN,
        private=False,
    )

    # Cleanup
    if TMP_CORPUS_FILE.exists():
        os.remove(TMP_CORPUS_FILE)
    for ext in [".model", ".vocab"]:
        path = Path(f"{SPM_MODEL_PREFIX}{ext}")
        if path.exists():
            path.unlink()

    log.info("🧹 Workspace cleaned up successfully.")
    log.info(f"🏁 Complete execution pipeline wrapped in {timedelta(seconds=int(time.time() - t0))}")

if __name__ == "__main__":
    main()