# We will clean the hh_rlhf data while removing the suffixes that abruptly end with a colon.
import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
from datasets import Dataset, load_dataset
from utils.utils import save_in_jsonl, distinctness, load_from_pickle
import os
def main():
    data_dir = "data/hh_train_len2"
    data_file = "train.json"
    logging.info(f"Loading train dataset from {data_dir}/{data_file}")
    train_dataset = load_dataset("json", data_dir = data_dir, data_files = data_file, streaming=False, split="train")
    logging.info(f"Loaded train dataset with {len(train_dataset)} samples")
    # Get the good and bad suffixes
    good_suffixes = [datum['suffix'][0] for datum in train_dataset]
    bad_suffixes = [datum['suffix'][1] for datum in train_dataset]
    # Find the number of good and bad suffixes that end with a colon
    good_suffixes_with_colon = [suffix for suffix in good_suffixes if suffix.endswith(":")]
    bad_suffixes_with_colon = [suffix for suffix in bad_suffixes if suffix.endswith(":")]
    logging.info(f"Number of good suffixes that end with a colon: {len(good_suffixes_with_colon)} vs bad suffixes: {len(bad_suffixes_with_colon)}")
    # Find the indices of the good and bad suffixes that end with a colon
    good_suffixes_with_colon_indices = [i for i, suffix in enumerate(good_suffixes) if suffix.endswith(":")]
    bad_suffixes_with_colon_indices = [i for i, suffix in enumerate(bad_suffixes) if suffix.endswith(":")]
    assert len(good_suffixes_with_colon_indices) == len(good_suffixes_with_colon)
    assert len(bad_suffixes_with_colon_indices) == len(bad_suffixes_with_colon)
    merged_indices = set(good_suffixes_with_colon_indices + bad_suffixes_with_colon_indices)
    # Remove all the instances from merged_indices
    cleaner_train_data = [train_dataset[i] for i in range(len(train_dataset)) if i not in merged_indices]
    assert len(cleaner_train_data) == len(train_dataset) - len(merged_indices)
    # Save the cleaner_train_data
    save_file = os.path.join(data_dir, "cleaner_train.json")
    save_in_jsonl(cleaner_train_data, save_file)
    logging.info(f"Saved {len(cleaner_train_data)} samples to {save_file}")
    # find suffixes that contain a colon
    # cleaner_colon_suffixes = [datum['suffix'][0] for datum in cleaner_train_data if ":" in datum['suffix'][0]]



if __name__ == "__main__":
    main()
