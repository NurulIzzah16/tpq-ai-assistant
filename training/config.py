"""
TPQ AI Assistant - Training Configuration

All training hyperparameters and model settings are centralized here.
Modify these values to experiment with different configurations.
"""

# ============================================================
# Model Configuration
# ============================================================

# Base model from Hugging Face / Unsloth
# Qwen2.5-1.5B-Instruct: lightweight instruct model, ideal for limited GPU
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"

# Maximum sequence length for tokenization
# Longer sequences use more VRAM; 2048 is sufficient for short Q&A pairs
MAX_SEQ_LENGTH = 2048

# Load model in 4-bit quantization to reduce VRAM usage
# Enables fine-tuning on GPUs with 8-16 GB VRAM (e.g., T4, RTX 3060)
LOAD_IN_4BIT = True

# ============================================================
# LoRA Configuration
# ============================================================

# LoRA rank: controls the number of trainable parameters
# Higher = more capacity but more VRAM; 16 is a good default
LORA_R = 16

# LoRA alpha: scaling factor for LoRA weights
# Typically set equal to LORA_R for stable training
LORA_ALPHA = 16

# LoRA dropout: regularization to prevent overfitting
# 0 is recommended by Unsloth for best performance
LORA_DROPOUT = 0

# Target modules for LoRA adaptation
# All linear layers in the transformer for comprehensive fine-tuning
LORA_TARGET_MODULES = [
    "q_proj",   # Query projection
    "k_proj",   # Key projection
    "v_proj",   # Value projection
    "o_proj",   # Output projection
    "gate_proj", # MLP gate projection
    "up_proj",   # MLP up projection
    "down_proj", # MLP down projection
]

# ============================================================
# Training Hyperparameters
# ============================================================

# Learning rate for AdamW optimizer
# 2e-4 is standard for LoRA fine-tuning
LEARNING_RATE = 2e-4

# Batch size per GPU device
# Reduce if running out of VRAM
PER_DEVICE_TRAIN_BATCH_SIZE = 2

# Number of gradient accumulation steps
# Effective batch size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 4

# Number of training epochs
# 3 epochs is typical for small datasets
NUM_TRAIN_EPOCHS = 3

# Number of warmup steps for learning rate scheduler
# Gradually increases LR from 0 to prevent early instability
WARMUP_STEPS = 5

# Weight decay for regularization
WEIGHT_DECAY = 0.01

# ============================================================
# Output Configuration
# ============================================================

# Directory to save the fine-tuned LoRA adapter
OUTPUT_DIR = "./models/qwen-tpq-sft"

# Logging steps interval
LOGGING_STEPS = 10

# Save checkpoint every N steps
SAVE_STEPS = 50

# Random seed for reproducibility
SEED = 42

# ============================================================
# Data Paths
# ============================================================

RAW_DATASET_PATH = "./data/raw/dataset.jsonl"
TRAIN_DATASET_PATH = "./data/train.jsonl"
VALIDATION_DATASET_PATH = "./data/validation.jsonl"
TEST_DATASET_PATH = "./data/test.jsonl"
