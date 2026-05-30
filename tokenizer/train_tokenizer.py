"""
train_unigram_tokenizer.py
──────────────────────────
Trains a 16K SentencePiece-equivalent (Unigram) tokenizer natively in Python/Rust.
Preserves exact normalization rules and regex pre-tokenization boundaries without high RAM overhead.
"""

import os
import re
import sys
import time
import unicodedata
import logging
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.normalizers import NFC, Strip, Replace, Sequence as NormSequence
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import UnigramTrainer
from transformers import PreTrainedTokenizerFast

import config as C

# ─────────────────────────────────────────────────────────────
# 0. Setup & Silence Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mathformer_unigram")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    log.error("Missing HF_TOKEN environment variable.")
    sys.exit(1)

OUTPUT_DIR = Path("./mathformer_16k_unigram")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# 2. Batch Streaming Iterator
# ─────────────────────────────────────────────────────────────
from tqdm import tqdm

class BatchIterator:
    def __init__(self, token: str, word_cap: int, batch_size: int = 1_000):
        self.token = token
        self.word_cap = word_cap
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[str]]:
        word_count = 0
        current_batch = []
        
        # Approximate how many batches total we will stream for the progress bar
        estimated_total_batches = self.word_cap // self.batch_size
        
        ds = load_dataset(
            C.NEMOTRON_DATASET, C.NEMOTRON_SUBSET,
            split=C.NEMOTRON_SPLIT, streaming=True, token=self.token
        ).skip(C.NEMOTRON_EVAL_ROWS)

        # Instantiate a clean, customizable progress bar layout
        pbar = tqdm(
            total=estimated_total_batches, 
            desc="📥 Streaming & Processing Batches", 
            unit="batch",
            dynamic_ncols=True
        )

        for row in ds:
            if word_count >= self.word_cap:
                if current_batch: 
                    yield current_batch
                    pbar.update(1)
                pbar.close()
                return
            
            text = normalize_text(row.get(C.NEMOTRON_TEXT_COL, "") or "")
            if len(text) > 50:
                words_in_row = len(text.split())
                word_count += words_in_row
                current_batch.append(text)
                
                if len(current_batch) >= self.batch_size:
                    yield current_batch
                    current_batch = []
                    
                    # Update progress bar and show absolute running word count
                    pbar.update(1)
                    pbar.set_postfix({"processed_words": f"{word_count:,}"})
                    
        if current_batch:
            yield current_batch
            pbar.update(1)
            
        pbar.close()

# ─────────────────────────────────────────────────────────────
# 3. Core Engine Build
# ─────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("🚀 Instantiating Unigram Tokenizer structural framework...")

    # Initialize empty Unigram model instead of BPE
    tokenizer = Tokenizer(Unigram())

    # 1. YOUR EXACT NORMALIZATION SET
    tokenizer.normalizer = NormSequence([
        NFC(), Strip(),
        Replace("\u00a0", " "), Replace("\u2009", " "),
        Replace("\u202f", " "), Replace("\u2003", " "),
        Replace("\u2002", " "), Replace("\u200b", ""), Replace("\ufeff", ""),
    ])

    # 2. YOUR EXACT REGEX PRE-TOKENIZATION RULES
    tokenizer.pre_tokenizer = Split(
        pattern=C.PRETOKENIZER_REGEX,
        behavior="isolated"
    )

    # 3. Configure the Unigram Trainer
    # Unigram automatically uses SentencePiece's optimization logic under the hood
    trainer = UnigramTrainer(
        vocab_size=C.VOCAB_SIZE,
        special_tokens=C.SPECIAL_TOKENS,
        show_progress=True,
        unk_token=C.UNK_TOKEN,
    )

    # Stream text through your exact normalization/regex parameters
    log.info(f"📥 Streaming data from {C.NEMOTRON_DATASET} directly through Rust pipes...")
    iterator = BatchIterator(token=HF_TOKEN, word_cap=C.TRAIN_WORD_CAP)
    
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    log.info("✓ Core model training complete.")

    # 4. Wrap into Fast Tokenizer Class
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

    # Save outputs locally
    fast_tok.save_pretrained(str(OUTPUT_DIR))
    log.info(f"💾 Fast tokenizer config saved to: {OUTPUT_DIR}")

    # Push to Hub
    log.info(f"📦 Synchronizing assets with Hugging Face Hub: {C.HUB_REPO_ID}")
    fast_tok.push_to_hub(
        C.HUB_REPO_ID,
        commit_message="Deploy safe 16K native Unigram tokenizer directly from stream.",
        token=HF_TOKEN,
        private=False
    )
    log.info(f"🏁 Process complete in {timedelta(seconds=int(time.time() - t0))}")

if __name__ == "__main__":
    main()