from pathlib import Path
import torch

# Get the directory where this script is located
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "wikitext-103"

# Tokenized outputs
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

# Dataset
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"

# Tokenizer

TOKENIZER_NAME = "gpt2"
TOKEN_DTYPE = "uint16"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'