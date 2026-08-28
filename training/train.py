"""
TPQ AI Assistant - Training Script

Fine-tunes Qwen2.5-1.5B-Instruct using Unsloth + SFT + LoRA.

Pipeline:
    1. Load Qwen base model (4-bit quantized via Unsloth)
    2. Apply LoRA adapters to all linear layers
    3. Format dataset using chat template
    4. Train using SFTTrainer
    5. Save LoRA adapter

Requirements:
    - NVIDIA GPU with CUDA support
    - pip install -r requirements-training.txt

Usage:
    python training/train.py
"""

import os
import sys
import json
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4BIT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LEARNING_RATE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_TRAIN_EPOCHS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    OUTPUT_DIR,
    LOGGING_STEPS,
    SAVE_STEPS,
    SEED,
    TRAIN_DATASET_PATH,
    VALIDATION_DATASET_PATH,
)


def load_dataset_from_jsonl(filepath):
    """Load dataset from JSONL file."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_chat_example(example, tokenizer):
    """
    Format a single chat example using the tokenizer's chat template.

    Converts the messages list into the model's expected chat format
    (e.g., ChatML for Qwen models).
    """
    messages = example["messages"]
    # Apply the tokenizer's built-in chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    print("=" * 60)
    print("TPQ AI Assistant - Training")
    print("=" * 60)
    print()

    # ================================================================
    # Step 1: Load base model with Unsloth
    # ================================================================
    print("[1/5] Loading base model with Unsloth...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Load in 4-bit: {LOAD_IN_4BIT}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect: float16 or bfloat16
        load_in_4bit=LOAD_IN_4BIT,
    )
    print("  Model loaded successfully!")
    print()

    # ================================================================
    # Step 2: Apply LoRA adapters
    # ================================================================
    print("[2/5] Applying LoRA adapters...")
    print(f"  Rank (r): {LORA_R}")
    print(f"  Alpha: {LORA_ALPHA}")
    print(f"  Dropout: {LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized checkpointing
        random_state=SEED,
    )

    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    print()

    # ================================================================
    # Step 3: Prepare dataset
    # ================================================================
    print("[3/5] Preparing dataset...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(project_root, TRAIN_DATASET_PATH.lstrip("./"))
    val_path = os.path.join(project_root, VALIDATION_DATASET_PATH.lstrip("./"))

    # Load datasets
    train_examples = load_dataset_from_jsonl(train_path)
    val_examples = load_dataset_from_jsonl(val_path)
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Validation examples: {len(val_examples)}")

    # Format using chat template
    train_formatted = [format_chat_example(ex, tokenizer) for ex in train_examples]
    val_formatted = [format_chat_example(ex, tokenizer) for ex in val_examples]

    # Convert to Hugging Face Dataset
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    print(f"  Datasets formatted with chat template")
    print()

    # ================================================================
    # Step 4: Configure and run SFTTrainer
    # ================================================================
    print("[4/5] Configuring SFTTrainer...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"  Warmup steps: {WARMUP_STEPS}")

    from trl import SFTTrainer, SFTConfig

    # SFTConfig replaces TrainingArguments in modern TRL
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",  # Field name in formatted dataset
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        seed=SEED,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",  # Memory-efficient optimizer
        lr_scheduler_type="linear",
        report_to="none",  # Disable W&B / tensorboard
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Modern TRL API (replaces tokenizer=)
    )

    print()
    print("  Starting training...")
    print("-" * 60)
    trainer_stats = trainer.train()
    print("-" * 60)
    print(f"  Training completed!")
    print(f"  Training loss: {trainer_stats.training_loss:.4f}")
    print(f"  Training time: {trainer_stats.metrics['train_runtime']:.1f}s")
    print()

    # ================================================================
    # Step 5: Save the fine-tuned LoRA adapter
    # ================================================================
    print("[5/5] Saving fine-tuned model...")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Save LoRA adapter and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("  Model and tokenizer saved successfully!")
    print()
    print("=" * 60)
    print("Training completed!")
    print()
    print(f"  Model saved to: {OUTPUT_DIR}")
    print(f"  To use the model:")
    print(f"    python inference/chat.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
