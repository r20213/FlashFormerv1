"""
config.py — Central configuration for MathFormer tokenizer training and evaluation.
All hyperparameters, paths, and constants live here. Import this everywhere.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Hub & Identity
# ---------------------------------------------------------------------------

HUB_REPO_ID        = "LastTransformer/MathFormer"
HUB_COMMIT_MESSAGE = "MathFormer v1 — 16K math-specialized BPE tokenizer"

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VOCAB_SIZE = 16_000          # Hard ceiling. BPE will stop at or before this.
MIN_FREQUENCY = 10            # Minimum pair frequency to merge.

# ---------------------------------------------------------------------------
# Training-corpus budget
# ---------------------------------------------------------------------------

TOK_TRAIN_TOKEN_BUDGET = 20_000_000    # 20 M tokens for tokenizer training
EVAL_TOKEN_BUDGET      =  1_000_000    # 1 M held-out tokens for evaluation

# Row-level sample caps (approximate — exact token count validated post-sample)
NEMOTRON_TRAIN_ROWS =  100_000   # ~120 tokens/row avg → ~72M tokens
DEEPMATH_TRAIN_ROWS =   50_000   # ~250 tokens/row avg → ~21M tokens
# Remaining budget filled by a second Nemotron pass if needed.

# Held-out eval rows (completely disjoint from training rows above)
NEMOTRON_EVAL_ROWS  =   10_000
DEEPMATH_EVAL_ROWS  =   5_000

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------

NEMOTRON_DATASET   = "nvidia/Nemotron-CC-Math-v1"
NEMOTRON_SUBSET    = "4plus"
NEMOTRON_SPLIT     = "train"
NEMOTRON_TEXT_COL  = "text"

DEEPMATH_DATASET   = "zwhe99/DeepMath-103K"
DEEPMATH_SPLIT     = "train"
DEEPMATH_TEXT_COL  = "question"   # primary text column
DEEPMATH_AUX_COLS  = ["r1_solution_1"]  # concatenated after question (sep = \n\n)

# ---------------------------------------------------------------------------
# Special tokens
# (ordered: [UNK, BOS, EOS, PAD, MASK, SEP, then domain specials])
# ---------------------------------------------------------------------------

SPECIAL_TOKENS: List[str] = [
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "<sep>",
    # --- reasoning scaffolding ---
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    # --- document structure ---
    "<question>",
    "</question>",
    "<solution>",
    "</solution>",
    "<step>",
    "</step>",
    # --- meta ---
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

# Token IDs are positionally fixed:  special_token_id = index in above list
UNK_TOKEN   = "<unk>"
BOS_TOKEN   = "<s>"
EOS_TOKEN   = "</s>"
PAD_TOKEN   = "<pad>"

# ---------------------------------------------------------------------------
# Normalisation (applied before pre-tokenisation)
# ---------------------------------------------------------------------------

# Unicode normalisation form. NFC keeps composed forms (é as one codepoint).
# We do NOT use NFKC because it collapses ligatures like ﬁ → fi which can
# alter LaTeX source in unpredictable ways.
UNICODE_NORM_FORM = "NFC"

# Strip leading/trailing whitespace from each document.
STRIP_WHITESPACE = True

# Collapse runs of 3+ blank lines to exactly 2 blank lines.
# Preserves paragraph structure without exploding token count.
COLLAPSE_BLANK_LINES = True

# Replace non-breaking spaces (U+00A0) and other exotic Unicode spaces with
# standard ASCII space. Prevents invisible vocabulary splits.
NORMALIZE_UNICODE_SPACES = True

# Whether to lowercase. False — math is case-sensitive (x ≠ X).
LOWERCASE = False

# ---------------------------------------------------------------------------
# Pre-tokenisation regex (Rust-compatible, passed to Regex pre-tokenizer)
#
# Design goals
# ────────────
# 1. Whitespace is ALWAYS its own isolated token. Every space/tab/newline
#    is split out BEFORE BPE sees any character, so BPE never learns
#    "word-with-leading-space" as a single unit. This is the opposite of
#    the GPT-2 Ġ-space convention and is intentional: it lets BPE focus
#    entirely on symbol sequences.
#
# 2. LaTeX commands (\frac, \sqrt, \alpha …) are a single atomic unit.
#    Regex: \\[a-zA-Z]+  matches backslash + letters.
#
# 3. Multi-digit numbers are shattered into individual digits so BPE
#    cannot merge "12" into one token. The digit rule (\d) is checked
#    BEFORE the alphanumeric "word" rule.
#
# 4. "3x" → ["3","x"]. Variable names and digits never share a token.
#    Achieved by separating digits (\d) from letters ([a-zA-Z_]).
#
# 5. Punctuation and math operators (+-*/=<>!^%&|~) are isolated as
#    single characters so BPE can optionally merge related operators
#    (e.g. <= might earn a merge) but never accidentally absorbs them
#    into a word.
#
# 6. Everything else (Unicode letters, accented chars) falls through to
#    the final catch-all.
#
# Alternation order matters: earlier branches take priority.
# ---------------------------------------------------------------------------

PRETOKENIZER_REGEX: str = (
    r"(\s+)"                  # 1. Whitespace — always isolated first
    r"|"
    r"(\\[a-zA-Z]+)"          # 2. LaTeX commands  e.g. \frac \alpha \sqrt
    r"|"
    r"(\d)"                   # 3. Single digit — shatters all numbers
    r"|"
    r"([a-zA-Z_][a-zA-Z_]*)"  # 4. Pure alphabetic/underscore runs (no digits)
    r"|"
    r"([+\-*/=<>!^%&|~@#$])"  # 5. Math / programming operators (one at a time)
    r"|"
    r"([{}()\[\],.:;?\\\"'`])"# 6. Brackets, punctuation
    r"|"
    r"(\S)"                   # 7. Catch-all: any non-whitespace Unicode char
)

# ---------------------------------------------------------------------------
# Paths (all relative to project root; Modal will override with volume mounts)
# ---------------------------------------------------------------------------

PROJECT_ROOT        = Path(__file__).parent
DATA_DIR            = PROJECT_ROOT / "data"
TOKENIZER_DIR       = PROJECT_ROOT / "tokenizer_output"
CORPUS_TRAIN_PATH   = DATA_DIR / "tok_train_corpus.txt"      # one doc per line
CORPUS_EVAL_PATH    = DATA_DIR / "tok_eval_corpus.txt"
TOKENIZER_JSON_PATH = TOKENIZER_DIR / "tokenizer.json"

# ---------------------------------------------------------------------------
# Modal-specific overrides (used when running inside Modal)
# ---------------------------------------------------------------------------

MODAL_VOLUME_MOUNT  = "/vol"
MODAL_DATA_DIR      = Path(MODAL_VOLUME_MOUNT) / "data"
MODAL_TOK_DIR       = Path(MODAL_VOLUME_MOUNT) / "tokenizer_output"
MODAL_CORPUS_TRAIN  = MODAL_DATA_DIR / "tok_train_corpus.txt"
MODAL_CORPUS_EVAL   = MODAL_DATA_DIR / "tok_eval_corpus.txt"
MODAL_TOK_JSON      = MODAL_TOK_DIR / "tokenizer.json"

# ---------------------------------------------------------------------------
# Evaluation — reference tokenizers to benchmark against
# ---------------------------------------------------------------------------

REFERENCE_TOKENIZERS: List[Tuple[str, str]] = [
    ("SmolLM2-135M",         "HuggingFaceTB/SmolLM2-135M"),
    ("Nemotron-Nano-9B",     "nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
    ("DeepSeekMath-7B-RL",   "deepseek-ai/deepseek-math-7b-rl"),
    ("MathCoder-L-13B",      "MathLLMs/MathCoder-L-13B"),
]

# ---------------------------------------------------------------------------
# Stress-test probe suite  (static / deterministic)
# ---------------------------------------------------------------------------

PROBE_DIGITS        = [str(d) for d in range(10)]
PROBE_NUMBERS       = [
    "0", "1", "42", "100", "1000", "999999",
    "3.14", "2.718", "-42", "-0.001",
    "1e5", "1e-5", "6.022e23",
    "0.333", "1234567890",
]
PROBE_LATEX_COMMANDS = [
    r"\frac", r"\sqrt", r"\int", r"\sum", r"\prod",
    r"\lim", r"\log", r"\sin", r"\cos", r"\tan",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon",
    r"\theta", r"\lambda", r"\mu", r"\pi", r"\sigma",
    r"\infty", r"\partial", r"\nabla", r"\forall", r"\exists",
    r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv",
    r"\implies", r"\iff", r"\in", r"\notin", r"\subset",
    r"\cup", r"\cap", r"\cdot", r"\times", r"\div",
]
PROBE_EXPRESSIONS = [
    r"\frac{3}{4}",
    r"\frac{x^2 + 1}{2x - 3}",
    r"x^2 + y^2 = r^2",
    r"e^{i\pi} + 1 = 0",
    r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
    r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
    r"3x + 4y = 12",
    r"\sqrt{2} \approx 1.41421356",
    r"\binom{n}{k} = \frac{n!}{k!(n-k)!}",
    r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
    r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
    r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}",
]
PROBE_CODE_SNIPPETS = [
    "def f(x):\n    return x**2 + 2*x + 1\n",
    "for i in range(10):\n    print(i * 3.14)\n",
    "import numpy as np\nx = np.array([1, 2, 3])\n",
    "result = (a + b) / (c - d)\n",
    "if x >= 0 and y <= 100:\n    z = x * y\n",
]
PROBE_MIXED = [
    "If x = 3 and y = 4, then \\sqrt{x^2 + y^2} = 5.",
    "Solve: \\frac{d}{dx}[x^3] = 3x^2",
    "The sum 1 + 2 + 3 + ... + n = \\frac{n(n+1)}{2}",
    "For n = 100: result = 100 * 101 / 2 = 5050",
]

# Operators whose standalone tokenisation we verify under Math Symbol Coverage
MATH_OPERATORS_CORPUS = [
    "+", "-", "*", "/", "=", "<", ">", "!", "^",
    "≤", "≥", "≠", "≈", "≡", "∈", "∉", "⊂",
    "∪", "∩", "∂", "∇", "∀", "∃", "∞", "π",
    "α", "β", "γ", "δ", "θ", "λ", "μ", "σ",
]
