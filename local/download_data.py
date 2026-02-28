from datasets import load_dataset
from pathlib import Path
from config import RAW_DATA_DIR, DATASET_NAME, DATASET_CONFIG

## lets load the wiki-103 dataset here

def download_and_save():
    RAW_DATA_DIR.mkdir(parents = True, exist_ok = True)
    ## check if data is already downloaded
    if(RAW_DATA_DIR / "train.txt").exists():
        print("Dataset already downloaded, skipping...")
        return
    ## else download from huggingface
    print("Downloading wikitext-103...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Save each split as plain text
    for split in ["train", "validation", "test"]:
        print(f"Saving {split} split...")
        output_file = RAW_DATA_DIR / f"{split}.txt"
        with open(output_file, "w") as f:
            for example in dataset[split]:
                f.write(example["text"])
                f.write("\n")  # Separator between documents

    # Print stats
    print(f"Dataset saved to {RAW_DATA_DIR}")
    print(f"Files: {list(RAW_DATA_DIR.glob('*.txt'))}")

if __name__ == "__main__":
    download_and_save()
