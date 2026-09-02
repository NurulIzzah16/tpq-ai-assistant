"""
TPQ AI Assistant - Dataset Preparation

Loads the raw dataset, validates format, removes duplicates,
splits into train/validation/test sets, and saves the splits.

Usage:
    python training/prepare_dataset.py
"""

import json
import os
import random
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    RAW_DATASET_PATH,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
    TEST_DATASET_PATH,
    SEED,
)

# Split ratios
TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.10


def load_dataset(filepath):
    """Load dataset from JSONL file."""
    examples = []
    invalid_count = 0
    line_number = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_number += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARNING] Line {line_number}: Invalid JSON, skipping.")
                invalid_count += 1
                continue

            examples.append(data)

    print(f"Loaded {len(examples)} examples ({invalid_count} invalid lines skipped)")
    return examples


def validate_example(example, index):
    """Validate a single example has the correct format."""
    # Must have 'messages' key
    if "messages" not in example:
        print(f"  [WARNING] Example {index}: Missing 'messages' key, skipping.")
        return False

    messages = example["messages"]

    # Must have at least 2 messages (user + assistant)
    if len(messages) < 2:
        print(f"  [WARNING] Example {index}: Less than 2 messages, skipping.")
        return False

    # First message must be from user
    if messages[0].get("role") != "user":
        print(f"  [WARNING] Example {index}: First message not from 'user', skipping.")
        return False

    # Second message must be from assistant
    if messages[1].get("role") != "assistant":
        print(
            f"  [WARNING] Example {index}: Second message not from 'assistant', skipping."
        )
        return False

    # Messages must have non-empty content
    for msg in messages:
        if not msg.get("content", "").strip():
            print(f"  [WARNING] Example {index}: Empty content found, skipping.")
            return False

    return True


def validate_dataset(examples):
    """Validate all examples in the dataset."""
    valid_examples = []

    for i, example in enumerate(examples):
        if validate_example(example, i + 1):
            valid_examples.append(example)

    removed = len(examples) - len(valid_examples)
    if removed > 0:
        print(f"Removed {removed} invalid examples")
    else:
        print("All examples are valid")

    return valid_examples


def remove_duplicates(examples):
    """Remove duplicate examples based on user message content."""
    seen = set()
    unique_examples = []

    for example in examples:
        # Create a fingerprint from the user message
        user_content = example["messages"][0]["content"].strip().lower()

        if user_content not in seen:
            seen.add(user_content)
            unique_examples.append(example)

    removed = len(examples) - len(unique_examples)
    if removed > 0:
        print(f"Removed {removed} duplicate examples")
    else:
        print("No duplicates found")

    return unique_examples


def split_dataset(examples, seed=42):
    """Split dataset into train/validation sets. Test set is kept separately."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * TRAIN_RATIO)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:]

    return train_data, val_data


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("TPQ AI Assistant - Dataset Preparation")
    print("=" * 60)
    print()

    # Step 1: Load raw dataset
    print("[1/5] Loading raw dataset...")
    raw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        RAW_DATASET_PATH.lstrip("./"),
    )
    examples = load_dataset(raw_path)
    print()

    # Step 2: Validate format
    print("[2/5] Validating dataset format...")
    examples = validate_dataset(examples)
    print()

    # Step 3: Remove duplicates
    print("[3/5] Removing duplicates...")
    examples = remove_duplicates(examples)
    print()

    # Step 4: Split dataset
    print("[4/4] Splitting dataset (90/10)...")
    train_data, val_data = split_dataset(examples, seed=SEED)
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    print()

    # Step 5: Save splits
    print("[5/5] Saving splits...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(project_root, TRAIN_DATASET_PATH.lstrip("./"))
    val_path = os.path.join(project_root, VALIDATION_DATASET_PATH.lstrip("./"))

    save_jsonl(train_data, train_path)
    print(f"  Saved: {train_path}")

    save_jsonl(val_data, val_path)
    print(f"  Saved: {val_path}")

    print()
    print("=" * 60)
    print("Dataset preparation completed.")
    print()
    print(f"  Total: {len(examples)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
