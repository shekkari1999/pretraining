
from config import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR,TRAIN_BIN, VAL_BIN, TOKEN_DTYPE, TOKENIZER_NAME
import tiktoken
import numpy as np
import time

def tokenize_and_save(input_path, output_path, enc):
    ## read raw text
    print(f'Reading {input_path.name}...')
    with open(input_path, "r") as f:
        text = f.read()
    print(f'Raw text size is: {len(text) / 1e6: .1f} MB')
    ## tokenize
    print('Tokenizing ...')
    start = time.time()
    tokens = enc.encode(text)
    elapsed = time.time() - start
    ## stats
    tokens_per_sec = len(tokens)/elapsed
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Tokenization speed:{tokens_per_sec:,.0f} tokens/sec")
    print(f"  Time: {elapsed:.1f}s")
    ## save as binary
    tokens = np.array(tokens, dtype = TOKEN_DTYPE)
    tokens.tofile(output_path)
    file_size_mb = output_path.stat().st_size /(1024 * 1024)
    print(f"  Saved to {output_path.name}({file_size_mb:.1f} MB)")

    return len(tokens)

def main():
    # Initialize tokenizer
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    print(f"Tokenizer: {TOKENIZER_NAME} (vocabsize: {enc.n_vocab})")

    # Tokenize train split
    train_count = tokenize_and_save(RAW_DATA_DIR / "train.txt", TRAIN_BIN, enc)

    # Tokenize validation split
    val_count = tokenize_and_save(RAW_DATA_DIR / "validation.txt", VAL_BIN, enc)

    # Summary
    print(f"\n--- Summary ---")
    print(f"Train tokens: {train_count:,}")
    print(f"Val tokens:   {val_count:,}")
    print(f"Total tokens: {train_count + val_count:,}")
    print(f"Train/Val ratio: {train_count/val_count:.1f}x")

if __name__ == "__main__":
      main()