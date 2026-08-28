"""
TPQ AI Assistant - Model Loader

Provides functions to load the fine-tuned Qwen model and generate responses.
Used by both the CLI chat and the FastAPI backend.

Usage:
    from inference.model_loader import load_model, generate_response
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# System prompt that guides the model's behavior
SYSTEM_PROMPT = (
    "Kamu adalah TPQ AI Assistant, asisten virtual untuk Taman Pendidikan Al-Quran (TPQ). "
    "Tugasmu adalah membantu menjawab pertanyaan terkait administrasi TPQ seperti "
    "pendaftaran santri, absensi, nilai, pembayaran SPP, jadwal belajar, dan informasi umum TPQ. "
    "Jawab dengan bahasa Indonesia yang sopan, jelas, dan ringkas. "
    "Jika pertanyaan di luar konteks administrasi TPQ, sampaikan dengan sopan bahwa "
    "kamu hanya dapat membantu pertanyaan terkait TPQ."
)

# Default generation parameters
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def load_model(model_path=None, model_name=None):
    """
    Load the fine-tuned model and tokenizer.

    Args:
        model_path: Path to the fine-tuned LoRA adapter directory.
                    If None, uses MODEL_PATH from config or env.
        model_name: Base model name (used if model_path doesn't exist
                    to fall back to the base model for testing).

    Returns:
        tuple: (model, tokenizer)
    """
    from training.config import MODEL_NAME, OUTPUT_DIR

    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", OUTPUT_DIR)

    if model_name is None:
        model_name = os.environ.get("MODEL_NAME", MODEL_NAME)

    # Check if fine-tuned model exists
    adapter_config = os.path.join(model_path, "adapter_config.json")
    use_finetuned = os.path.exists(adapter_config)

    if use_finetuned:
        print(f"Loading fine-tuned model from: {model_path}")
        load_path = model_path
    else:
        print(f"Fine-tuned model not found at: {model_path}")
        print(f"Falling back to base model: {model_name}")
        load_path = model_name

    try:
        # Try loading with Unsloth (fastest, GPU required)
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        # Set model to inference mode
        FastLanguageModel.for_inference(model)
        print("Model loaded with Unsloth (GPU accelerated)")

    except (ImportError, Exception) as e:
        print(f"Unsloth not available ({e}), using standard Transformers...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            load_path if use_finetuned else model_name,
            trust_remote_code=True,
        )

        if use_finetuned:
            # Load base model + LoRA adapter
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model.eval()
        print("Model loaded with Transformers")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    user_message,
    system_prompt=None,
    max_new_tokens=None,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
):
    """
    Generate a response from the model.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        user_message: The user's input message.
        system_prompt: Optional custom system prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (higher = more creative).
        top_p: Nucleus sampling parameter.

    Returns:
        str: The model's generated response.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    if max_new_tokens is None:
        max_new_tokens = int(
            os.environ.get("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
        )

    # Build the messages list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generate
    model.generation_config.max_length = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip the input)
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()
