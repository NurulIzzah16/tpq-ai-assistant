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
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# System prompt that guides the model's behavior
SYSTEM_PROMPT = (
    "Kamu adalah TPQ AI Assistant. Tugasmu HANYA menjawab pertanyaan terkait "
    "administrasi dan informasi TPQ Mambaus Sholihin.\n\n"
    "OFFICIAL TPQ WEBSITE KNOWLEDGE is the authoritative source for all information about the TPQ website, menus, features, roles, permissions, and system functionality.\n\n"
    "ATURAN MENJAWAB:\n"
    "1. Only answer using verified information contained in OFFICIAL TPQ WEBSITE KNOWLEDGE.\n"
    "2. Never invent menu names. ONLY use exact menu names from the knowledge.\n"
    "3. Never invent submenu names.\n"
    "4. Never invent navigation paths. NEVER combine unrelated menus.\n"
    "5. Never create menu names from feature names. NEVER assume a feature belongs to another menu.\n"
    "6. Never expose raw URLs/paths in normal AI responses unless explicitly asked.\n"
    "7. Never invent permissions. If an exact permission/action is NOT VERIFIED, DO NOT say 'Anda bisa mengedit/menambahkan/menghapus/mengklik' dsb.\n"
    "8. Never claim a feature does not exist if the feature is verified in OFFICIAL TPQ WEBSITE KNOWLEDGE.\n"
    "9. If a feature exists but the exact navigation/menu is not verified, say that the feature is available but do not invent where it is located.\n"
    "10. If the website explicitly shows the menu name, use the EXACT menu name from the knowledge file.\n"
    "11. Do not replace an exact website menu name with a similar invented name.\n"
    "12. Do not assume that two related features are located under the same menu.\n"
    "13. Do not claim 'real-time' unless the website explicitly verifies real-time behavior.\n"
    "14. Do not invent procedures or steps that were not verified.\n"
    "15. If the requested information is not present in the official knowledge, answer: "
    "'Informasi tersebut belum tersedia dalam data saya. Silakan hubungi admin TPQ.' Do not guess.\n"
    "16. Jika pertanyaan di luar konteks TPQ, jawab persis: 'Maaf, saya hanya dapat membantu pertanyaan terkait administrasi dan informasi TPQ Mambaus Sholihin.'\n"
    "17. Jawab dengan singkat, jelas, dan menggunakan bahasa Indonesia yang sopan."
)

def load_official_knowledge():
    knowledge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge", "tpq_website_knowledge.json")
    try:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        knowledge = "=== OFFICIAL TPQ WEBSITE KNOWLEDGE ===\n"
        
        gen = data.get("general_information", {})
        knowledge += "GENERAL INFORMATION:\n"
        for k, v in gen.items():
            knowledge += f"- {k.replace('_', ' ').title()}: {v}\n"
            
        roles = data.get("roles", {})
        knowledge += "\nROLES AND FEATURES:\n"
        for role, info in roles.items():
            knowledge += f"ROLE: {role.upper()}\n"
            knowledge += f"  MENUS: {', '.join(info.get('menus', []))}\n"
            knowledge += f"  FEATURES: {', '.join(info.get('features', []))}\n"
            knowledge += f"  PERMISSIONS: {', '.join(info.get('permissions', []))}\n"
            
        return knowledge + "====================================\n\n"
    except Exception as e:
        return ""

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

    # Simple OOD (Out-of-Domain) detection
    user_message_lower = user_message.lower()
    tpq_keywords = [
        "spp", "absen", "hadir", "izin", "sakit", "alfa",
        "nilai", "rapor", "hafal", "prestasi", "daftar",
        "santri", "wali", "ustadz", "guru", "admin", "pengumuman", "jadwal", "libur",
        "tabung", "profil", "bayar", "tpq", "mambaus", "sholihin", "kepala", "umi latifah",
        "alamat", "telepon", "kontak", "biaya", "login", "password", "akun",
        "sistem", "juz", "laporan", "rekap", "kelas", "anak", "transfer", "rekening",
        "data", "usia", "umur"
    ]
    
    greetings = ["halo", "hai", "assalamualaikum", "selamat", "pagi", "siang", "sore", "malam", "test"]
    
    is_in_domain = False
    
    # Check if contains any TPQ keywords
    for kw in tpq_keywords:
        if kw in user_message_lower:
            is_in_domain = True
            break
            
    # Check if it's a short greeting message
    if not is_in_domain and len(user_message_lower.split()) <= 3:
        for g in greetings:
            if g in user_message_lower:
                is_in_domain = True
                break
                
    if not is_in_domain:
        return "Maaf, saya hanya dapat membantu pertanyaan terkait administrasi dan informasi TPQ Mambaus Sholihin."

    # Load official knowledge
    official_knowledge = load_official_knowledge()
    final_system_prompt = system_prompt
    if official_knowledge:
        final_system_prompt = final_system_prompt + "\n\n" + official_knowledge

    # Build the messages list
    messages = [
        {"role": "system", "content": final_system_prompt},
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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip the input)
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()
