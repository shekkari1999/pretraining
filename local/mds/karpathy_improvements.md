# Karpathy's prepare.py vs Our Implementation — Key Differences

Reference: nanoGPT `data/openwebtext/prepare.py`

## 1. `encode_ordinary()` vs `encode()` [Medium Priority]

Karpathy uses `encode_ordinary()` which ignores special tokens during encoding.
We use `encode()` which may handle special tokens differently.

```python
# Karpathy
ids = enc.encode_ordinary(example['text'])

# Ours
tokens = enc.encode(text)
```

For training data, `encode_ordinary()` is preferred — you control special tokens yourself.

## 2. End-of-Text (EOT) Token After Every Document [HIGH Priority]

Karpathy appends `enc.eot_token` (token ID 50256) after each document.
We don't add any document boundary markers.

```python
# Karpathy
ids = enc.encode_ordinary(example['text'])
ids.append(enc.eot_token)  # 50256
```

Without EOT:
- Model can't learn document boundaries
- Thinks document A flows into document B as one continuous text
- Model never learns "this is where one document ends"

## 3. Document-Level Tokenization vs Concatenated Text [HIGH Priority]

Karpathy tokenizes each document individually, preserving boundaries.
We concatenate all text into one giant string, then tokenize — losing document boundaries.

```python
# Karpathy: per-document
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

tokenized = split_dataset.map(process, num_proc=num_proc)
```

```python
# Ours: one big blob
text = f.read()           # entire file as one string
tokens = enc.encode(text) # tokenize everything at once
```

## 4. Parallel Tokenization with `.map()` [Low Priority for wikitext-103]

Karpathy uses HuggingFace `.map()` with `num_proc=8` for parallel tokenization across CPU cores.
We tokenize single-threaded.

```python
# Karpathy
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    num_proc=num_proc,  # 8 workers
)
```

Not needed for wikitext-103 (~550MB) but essential for larger datasets like OpenWebText (~54GB).

## 5. Memory-Efficient Writing with `np.memmap` [Low Priority for wikitext-103]

Karpathy writes tokens to disk via `np.memmap` in batches — never holds all tokens in RAM.
We hold all tokens in memory then dump with `.tofile()`.

```python
# Karpathy: stream to disk
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
for batch_idx in tqdm(range(total_batches)):
    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
    arr_batch = np.concatenate(batch['ids'])
    arr[idx : idx + len(arr_batch)] = arr_batch
arr.flush()

# Ours: all in memory
tokens = np.array(tokens, dtype=TOKEN_DTYPE)
tokens.tofile(output_path)
```

For wikitext-103 (~103M tokens × 2 bytes = ~200MB) ours works fine.
For OpenWebText (~9B tokens × 2 bytes = ~17GB) you'd run out of RAM.

## Action Items

- [ ] Switch `enc.encode()` to `enc.encode_ordinary()`
- [ ] Add EOT token after each document
- [ ] Restructure to tokenize per-document instead of as one blob
- [ ] (Later) Add parallel `.map()` when scaling to larger datasets
- [ ] (Later) Use `np.memmap` for writing when dataset exceeds RAM
