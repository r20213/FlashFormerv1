"""
config.py
─────────
Unified configuration and regex operational specifications for MathFormer Tokenizer pipeline.
"""

from pathlib import Path

# Seed configurations
RANDOM_SEED = 42
UNICODE_NORM_FORM = "NFC"

# Resource & Target Capacities
VOCAB_SIZE = 16_000
MIN_FREQUENCY = 20
TRAIN_WORD_CAP = 100_000_000  # ~150M tokens allocation overhead

# Dataset Stream Targets
NEMOTRON_DATASET = "nvidia/Nemotron-CC-Math-v1"
NEMOTRON_SUBSET = "4plus"
NEMOTRON_SPLIT = "train"
NEMOTRON_TEXT_COL = "text"
NEMOTRON_EVAL_ROWS = 8_000  # Held out validation size metrics allocation

DEEPMATH_DATASET = "zwhe99/DeepMath-103K"
DEEPMATH_SPLIT = "train"
DEEPMATH_TEXT_COL = None  # Multiple column concatenation strategy for question, solution, and final answer fields
DEEPMATH_AUX_COLS = ["question", "r1_solution_1","final_answer"]
DEEPMATH_EVAL_ROWS = 2_000

# Directory Resolution Ecosystem Locations
MODAL_VOLUME_MOUNT = "/vol/shared"
MODAL_DATA_DIR = Path("/vol/shared/data")
MODAL_TOK_DIR = Path("/vol/shared/tokenizer")
MODAL_TOK_JSON = MODAL_TOK_DIR / "tokenizer.json"

DATA_DIR = Path("./data")
TOKENIZER_DIR = Path("./mathformer_tokenizer")
TOKENIZER_JSON_PATH = TOKENIZER_DIR / "tokenizer.json"

# Core Special Tokens Array Map Structure Definitions
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, "<|step|>", "<|source|>","<think>","</think>","<answer>","</answer>"]

# Pretokenization Split Expression Pipeline Lookups
# Priority Breakdown Strategy:
# 1. Capture and isolate multi-character horizontal/vertical whitespace spacing rules cleanly.
# 2. Prevent structural LaTeX string primitives matching from splitting across tokens (`\alpha`).
# 3. Mandate structural explicit isolation individual token splitting for numbers (`1`, `2`, `3`).
# 4. Standard contiguous letters tracking blocks.
PRETOKENIZER_REGEX = (
    r"(?:\s+)|"
    r"(?:\\(?:[a-zA-Z]+|[^a-zA-Z]))|"
    r"(?:[0-9])|"
    r"(?:[a-zA-Z]+)|"
    r"(?:[^\s\w])"
)

# Benchmark Baselining Comparative Configurations
REFERENCE_TOKENIZERS = [
    ("DeepSeek-Math-V2", "deepseek-ai/DeepSeek-Math-V2"),
    ("Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-1.5B"),
    ("NVIDIA-Nemotron-Nano-9B-v2", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
    ("SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct")
]

# Evaluation Probes Data Array Specifications
MATH_OPERATORS_CORPUS = ["+", "-", "×", "÷", "=", "≠", "≤", "≥", "±", "∑", "∏", "∫", "∂", "∇"]
PROBE_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
PROBE_NUMBERS = ["3.14159", "1000000", "42", "1234", "0.001", "999"]
PROBE_LATEX_COMMANDS = ["\\frac", "\\sqrt", "\\alpha", "\\beta", "\\int", "\\matrix", "\\infty", "\\times"]
PROBE_EXPRESSIONS = [
    "f(x) = \\sin(x) + \\frac{1}{2}\\ln(x)",
    "\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}",
    "x_{1,2} = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
]

# Deployment Targets
HUB_REPO_ID = "LastTransformer/MathFormer-16K-BPE"
HUB_COMMIT_MESSAGE = "Optimize memory-efficient training workflow layout."