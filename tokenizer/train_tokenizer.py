"""
train_unigram_file_based.py
───────────────────────────
Streams, normalizes, and writes Nemotron to a local flat file,
then triggers the file-backed Rust Unigram trainer to guarantee low RAM.
"""

import os
import re
import sys
import time
import unicodedata
import logging
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.normalizers import NFC, Strip, Replace, Sequence as NormSequence
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import UnigramTrainer
from transformers import PreTrainedTokenizerFast

import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mathformer_flat")

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
    
    # Pack the components together into a single structured block
    # Modify the markers below if you use specific custom tokens (e.g., <|thought|>)
    return f"Question: {question}\nThought: {thought}\nFinal Answer: {output}"

def build_local_corpus():
    if TMP_CORPUS_FILE.exists():
        log.info(f"💾 Found existing normalized corpus file at {TMP_CORPUS_FILE}. Skipping extraction.")
        return

    log.info(f"📝 Creating a balanced, normalized corpus file: {TMP_CORPUS_FILE}")
    word_count = 0
    
    # Split the word cap evenly between both datasets to get a balanced distribution
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

        for row in nemotron_ds:
            if nemotron_words >= target_per_dataset:
                break
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

        for row in deepmath_ds:
            if deepmath_words >= target_per_dataset or word_count >= C.TRAIN_WORD_CAP:
                break
            # Reconstruct the question/thought/output block structure natively
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
# 3. Stage 2: Train via File References
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    
    # Run data extraction step
    build_local_corpus()

    log.info("⚙️ Configuring Tokenizer structural rules...")
    tokenizer = Tokenizer(Unigram())

    # YOUR EXACT NORMALIZATION SET
    tokenizer.normalizer = NormSequence([
        NFC(), Strip(),
        Replace("\u00a0", " "), Replace("\u2009", " "),
        Replace("\u202f", " "), Replace("\u2003", " "),
        Replace("\u2002", " "), Replace("\u200b", ""), Replace("\ufeff", ""),
    ])

    # YOUR EXACT REGEX PRE-TOKENIZATION RULES
    tokenizer.pre_tokenizer = Split(
        pattern=C.PRETOKENIZER_REGEX,
        behavior="isolated"
    )

    trainer = UnigramTrainer(
        vocab_size=C.VOCAB_SIZE,
        special_tokens=C.SPECIAL_TOKENS,
        show_progress=True,
        unk_token=C.UNK_TOKEN,
        max_piece_length=16,
    )

    # HERE IS THE RAM PROTECTION SAVIOR:
    # Instead of an iterator, pass the file path as a string list.
    log.info("🔨 Launching file-streamed Rust Unigram Trainer...")
    tokenizer.train([str(TMP_CORPUS_FILE)], trainer=trainer)
    log.info("✓ Tokenizer training complete.")

    # Wrap to Fast Tokenizer
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=C.UNK_TOKEN, bos_token=C.BOS_TOKEN,
        eos_token=C.EOS_TOKEN, pad_token=C.PAD_TOKEN,
        model_max_length=4096, padding_side="right", truncation_side="right"
    )
    fast_tok.add_special_tokens({
        "additional_special_tokens": [
            t for t in C.SPECIAL_TOKENS 
            if t not in (C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN, C.PAD_TOKEN)
        ]
    })

    fast_tok.save_pretrained(str(OUTPUT_DIR))
    log.info(f"💾 Converted fast configuration assets saved to: {OUTPUT_DIR}")

    # Push to Hub
    log.info(f"🚀 Pushing final assets to Hub: {C.HUB_REPO_ID}")
    fast_tok.push_to_hub(
        C.HUB_REPO_ID,
        commit_message="Deploy file-backed 16K Unigram tokenizer safely.",
        token=HF_TOKEN,
        private=False
    )
    
    # Optional cleanup of the big text file
    if TMP_CORPUS_FILE.exists():
        os.remove(TMP_CORPUS_FILE)
        log.info("🧹 Cleaned up temporary local corpus file.")
        
    log.info(f"🏁 Complete execution pipeline wrapped in {timedelta(seconds=int(time.time() - t0))}")

if __name__ == "__main__":
    main()