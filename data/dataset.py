"""
data/dataset.py — Streaming data pipeline for pretraining

Pipeline overview:
  1. HuggingFace datasets.interleave_datasets() in streaming mode, weighted
     by the probabilities in DataConfig.sources.
  2. Per-row text extraction (handles UltraInteract special-case formatting).
  3. Inline filtering: length gate (min/max tokens), language confidence
     (fasttext), toxicity score (Detoxify).
  4. Tokenisation with SentencePiece (loaded once, shared across workers).
  5. Sequence packing: token IDs are accumulated in a ring buffer and sliced
     into fixed-length chunks of `seq_len` with EOS as document separator.
     No padding — every token in the batch is a real token.
  6. PackedDataset wraps the generator as a PyTorch IterableDataset and is
     passed to a standard DataLoader with num_workers > 0.

Memory and I/O notes:
  - All dataset access is streaming — no disk materialisation of the full corpus.
  - SentencePiece model is loaded once per worker (in worker_init_fn or lazily
    in __iter__) to avoid forking a loaded model.
  - Detoxify and fasttext models are loaded lazily and cached per process.
  - The generator is intentionally infinite (cycles via itertools.cycle on the
    interleaved stream) so the DataLoader never raises StopIteration mid-epoch.
    Training terminates by step count, not by dataset exhaustion.

Validation split:
  - A separate PackedDataset(split="validation") is constructed identically
    but seeds the interleave differently and draws from a fixed token budget
    (cfg.train.val_tokens). Used only for loss evaluation.
"""

from __future__ import annotations

import os
import random
import itertools
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

import sentencepiece as spm
from datasets import load_dataset, interleave_datasets


# ---------------------------------------------------------------------------
# Lazy singleton caches (per worker process)
# ---------------------------------------------------------------------------

_detoxify_model  = None
_fasttext_model  = None
_spm_model: Optional[spm.SentencePieceProcessor] = None


def _get_detoxify():
    global _detoxify_model
    if _detoxify_model is None:
        try:
            from detoxify import Detoxify
            _detoxify_model = Detoxify("original", device="cpu")
        except ImportError:
            _detoxify_model = None  # skip toxicity filter if not installed
    return _detoxify_model


def _get_fasttext(model_path: str):
    global _fasttext_model
    if _fasttext_model is None and model_path and os.path.exists(model_path):
        try:
            import fasttext
            fasttext.FastText.eprint = lambda *args, **kwargs: None  # silence stderr
            _fasttext_model = fasttext.load_model(model_path)
        except ImportError:
            _fasttext_model = None
    return _fasttext_model


def _get_spm(model_path: str) -> spm.SentencePieceProcessor:
    global _spm_model
    if _spm_model is None:
        _spm_model = spm.SentencePieceProcessor()
        _spm_model.Load(model_path)
    return _spm_model


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(row: dict, source: dict) -> Optional[str]:
    """Extract and lightly clean text from a dataset row."""
    col = source.get("text_column")

    if col is None:
        # UltraInteract: compose instruction + response
        if source["path"] == "openbmb/UltraInteract_sft":
            instruction = (row.get("instruction") or "").strip()
            response    = (row.get("response")    or "").strip()
            if not instruction or not response:
                return None
            return f"{instruction}\n\n{response}"
        return None

    text = row.get(col)
    if not text or not isinstance(text, str):
        return None
    return text.strip() or None


# ---------------------------------------------------------------------------
# Inline filters
# ---------------------------------------------------------------------------

def _passes_length_filter(
    token_ids:    list[int],
    min_tokens:   int,
    max_tokens:   int,
) -> bool:
    n = len(token_ids)
    return min_tokens <= n <= max_tokens


def _passes_lang_filter(text: str, fasttext_model, threshold: float) -> bool:
    if fasttext_model is None:
        return True  # skip filter if model not available
    try:
        labels, scores = fasttext_model.predict(text.replace("\n", " "), k=1)
        label = labels[0].replace("__label__", "")
        return label == "en" and scores[0] >= threshold
    except Exception:
        return True


def _passes_toxicity_filter(text: str, detoxify_model, threshold: float) -> bool:
    if detoxify_model is None:
        return True
    try:
        result = detoxify_model.predict(text)
        return result.get("toxicity", 0.0) < threshold
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _load_hf_source(source: dict, streaming: bool = True):
    """Load a single HuggingFace dataset source in streaming mode."""
    hf_token = os.environ.get("HF_TOKEN", None)
    kwargs = {
        "streaming":          streaming,
        "split":              source["split"],
        "trust_remote_code":  True,
        "token":              hf_token,
    }
    if source["path"] == "bigcode/starcoderdata":
        kwargs["data_dir"] = source["name"]
    elif source.get("name") is not None:
        kwargs["name"] = source["name"]

    return load_dataset(source["path"], **kwargs)


# ---------------------------------------------------------------------------
# Core token generator
# ---------------------------------------------------------------------------

def _token_stream(
    sources:             list[dict],
    tokenizer_path:      str,
    seq_len:             int,
    eos_token_id:        int,
    pad_id:              int,
    min_doc_tokens:      int,
    max_doc_tokens:      int,
    toxicity_threshold:  float,
    lang_threshold:      float,
    fasttext_model_path: str,
    seed:                int = 42,
) -> Iterator[list[int]]:
    """
    Infinite generator that yields packed token sequences of exactly `seq_len`.

    Packing strategy:
      - Maintain a buffer (list of token IDs).
      - For each document: tokenise, optionally filter, then append [tokens + EOS].
      - Whenever buffer length >= seq_len, yield buffer[:seq_len] and keep the rest.
      - Wraps around the dataset indefinitely via itertools.cycle.
    """
    sp           = _get_spm(tokenizer_path)
    detox        = _get_detoxify()
    ft_model     = _get_fasttext(fasttext_model_path)

    # Build streaming interleaved dataset
    hf_datasets  = [_load_hf_source(src) for src in sources]
    weights      = [src["weight"] for src in sources]
    # Normalise weights (interleave_datasets requires probabilities that sum to 1)
    total_w      = sum(weights)
    probs        = [w / total_w for w in weights]

    interleaved  = interleave_datasets(
        hf_datasets,
        probabilities=probs,
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    # Infinite cycling: restart the interleaved stream when exhausted
    stream       = itertools.cycle(interleaved)

    buffer: list[int] = []

    for row in stream:
        # Determine which source this row came from to get the text column
        # interleave_datasets preserves a __source__ column if available;
        # we match by sampling deterministically from the weighted sources.
        # Since interleave_datasets mixes rows, we use a source-agnostic extractor.
        text = _extract_text_generic(row, sources)
        if text is None:
            continue

        # Tokenise (no BOS prepended; BOS would confuse packing across documents)
        try:
            token_ids: list[int] = sp.Encode(text)
        except Exception:
            continue

        if not _passes_length_filter(token_ids, min_doc_tokens, max_doc_tokens):
            continue

        if not _passes_lang_filter(text, ft_model, lang_threshold):
            continue

        if not _passes_toxicity_filter(text, detox, toxicity_threshold):
            continue

        # Append tokens + EOS separator
        buffer.extend(token_ids)
        buffer.append(eos_token_id)

        # Yield packed chunks
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]


def _extract_text_generic(row: dict, sources: list[dict]) -> Optional[str]:
    """
    Try each source's extractor in order until one works.
    This is used when interleave_datasets doesn't expose which source a row came from.
    """
    # Fast path: try common column names
    for col in ("content", "text", "code"):
        val = row.get(col)
        if val and isinstance(val, str) and len(val.strip()) >= 40:
            return val.strip()

    # UltraInteract fallback
    instruction = row.get("instruction", "")
    response    = row.get("response",    "")
    if instruction and response:
        return f"{instruction.strip()}\n\n{response.strip()}"

    return None


# ---------------------------------------------------------------------------
# IterableDataset wrapper
# ---------------------------------------------------------------------------

class PackedDataset(IterableDataset):
    """
    PyTorch IterableDataset that wraps the packed token stream.

    Each item is a 1-D LongTensor of shape (seq_len,).
    The DataLoader collates a batch of these into (B, seq_len).

    Args:
        cfg:   Full Config object (DataConfig + TrainConfig fields used).
        split: "train" or "validation" (affects seed and token budget).
        max_tokens: If set, the generator yields at most this many tokens total
                    (used for the fixed-size validation shard).
    """

    def __init__(self, cfg, split: str = "train", max_tokens: Optional[int] = None):
        super().__init__()
        self.cfg         = cfg
        self.split       = split
        self.max_tokens  = max_tokens
        self.seed        = cfg.train.seed if split == "train" else cfg.train.seed + 1

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        seed        = self.seed + (worker_info.id if worker_info is not None else 0)

        dcfg = self.cfg.data
        gen  = _token_stream(
            sources             = dcfg.sources,
            tokenizer_path      = dcfg.tokenizer_path,
            seq_len             = dcfg.seq_len,
            eos_token_id        = dcfg.eos_token_id,
            pad_id              = 3,
            min_doc_tokens      = dcfg.min_doc_tokens,
            max_doc_tokens      = dcfg.max_doc_tokens,
            toxicity_threshold  = dcfg.toxicity_threshold,
            lang_threshold      = dcfg.lang_threshold,
            fasttext_model_path = getattr(dcfg, "fasttext_model_path", ""),
            seed                = seed,
        )

        tokens_yielded = 0
        for chunk in gen:
            if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                return
            yield torch.tensor(chunk, dtype=torch.long)
            tokens_yielded += len(chunk)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(cfg, split: str = "train") -> DataLoader:
    """
    Build a DataLoader for training or validation.

    For validation, max_tokens is set to cfg.train.val_tokens so the loader
    produces a fixed-size shard and then stops.

    For training, the dataset is infinite (no max_tokens).
    """
    max_tokens = cfg.train.val_tokens if split == "validation" else None
    dataset    = PackedDataset(cfg, split=split, max_tokens=max_tokens)

    return DataLoader(
        dataset,
        batch_size  = cfg.train.micro_batch_size,
        num_workers = cfg.data.num_proc,
        pin_memory  = True,
        # IterableDataset doesn't support shuffle (data is already randomised
        # by the interleave seed and HuggingFace shuffling).
        shuffle     = False,
        drop_last   = True,
    )
