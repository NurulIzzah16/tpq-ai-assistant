"""
TPQ AI Assistant - Model Loader

Provides functions to load the fine-tuned Qwen model and generate responses.
Used by both the CLI chat and the FastAPI backend.

Usage:
    from inference.model_loader import load_model, generate_response
"""

import contextlib
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


def get_verified_wali_response(user_message):
    """
    Deterministic responses for verified Wali Santri intents.

    Rules:
    - Verified Wali menus only.
    - Wali Santri is read-only based on verified audit.
    - Never invent menus, submenus, URLs, buttons, permissions,
      payment methods, or procedures.
    - Management/editing intents have higher priority than view intents.
    - Payment intent is separated from payment-history/view intent.
    - Multi-feature questions are handled before single-feature rules.
    """

    text = user_message.lower().strip()

    # ============================================================
    # HELPERS
    # ============================================================

    def contains_any(keywords):
        return any(keyword in text for keyword in keywords)

    def contains_all(keywords):
        return all(keyword in text for keyword in keywords)

    # ------------------------------------------------------------
    # Detect whether the user is asking to modify/manage data
    # ------------------------------------------------------------

    management_action = contains_any([
        "ubah",
        "mengubah",
        "edit",
        "mengedit",
        "perbaiki",
        "memperbaiki",
        "koreksi",
        "mengoreksi",
        "betulkan",
        "membetulkan",
        "ganti",
        "mengganti",
        "hapus",
        "menghapus",
        "tambah",
        "menambah",
        "tambahkan",
        "menambahkan",
        "input",
        "menginput",
        "masukkan",
        "memasukkan",
        "isi",
        "mengisi",
        "mencatat",
        "mencatatkan",
        "kelola",
        "mengelola",
        "mengatur",
        "atur",
    ])

    # ============================================================
    # 1. WALI VS ADMIN / AKSES PENGELOLAAN
    # ============================================================

    if contains_any([
        "seperti admin",
        "seperti akun admin",
        "sama seperti admin",
        "seperti administrator",
        "akses admin",
        "akses administrator",
        "hak akses admin",
        "hak akses administrator",
        "akun admin",
        "akun administrator",
        "mengelola data",
        "kelola data",
        "mengelola data anak",
        "kelola data anak",
        "mengatur data anak",
        "mengedit data anak",
        "mengubah data anak",
        "memperbaiki data anak",
        "menghapus data anak",
        "menambah data anak",
    ]):
        return (
            "Tidak. Akun Wali Santri digunakan untuk melihat informasi "
            "santri dan data terkait, seperti absensi, nilai dan rapor, "
            "hafalan Juz 30, buku prestasi, tabungan, status SPP, "
            "dan pengumuman."
        )

    # ============================================================
    # 2. PERTANYAAN TENTANG EDIT DATA ANAK SECARA UMUM
    # ============================================================

    if contains_any([
        "data anak",
        "profil anak",
        "informasi anak",
        "biodata anak",
    ]) and management_action:
        return (
            "Wali Santri dapat melihat informasi anak melalui akun "
            "Wali Santri. Informasi mengenai perubahan, penambahan, "
            "atau penghapusan data anak dari akun Wali Santri belum "
            "tersedia dalam data saya. Silakan hubungi admin TPQ."
        )

    # ============================================================
    # 3. SPP - PEMBAYARAN DARI MENU SPP
    # ============================================================

    if "spp" in text and contains_any([
        "bisa bayar",
        "bisa membayar",
        "bisa melakukan pembayaran",
        "bisa untuk bayar",
        "bisa untuk membayar",
        "untuk bayar",
        "untuk membayar",
        "digunakan untuk bayar",
        "digunakan untuk membayar",
        "digunakan untuk pembayaran",
        "bayar dari menu",
        "membayar dari menu",
        "pembayaran dari menu",
        "bayar melalui menu",
        "membayar melalui menu",
        "pembayaran melalui menu",
        "tombol bayar",
        "tombol pembayaran",
    ]):
        return (
            "Tidak. Pembayaran SPP dilakukan secara langsung di TPQ "
            "dan dicatat secara manual oleh admin. Menu SPP pada "
            "akun Wali Santri digunakan untuk melihat status dan "
            "riwayat pembayaran."
        )

    # ============================================================
    # 4. SPP - ONLINE / TRANSFER / QRIS
    # ============================================================

    if "spp" in text and contains_any([
        "online",
        "qris",
        "transfer",
        "transfer bank",
        "rekening",
        "nomor rekening",
        "mobile banking",
        "m-banking",
        "mbanking",
        "payment gateway",
        "e-wallet",
        "dompet digital",
        "ovo",
        "dana",
        "gopay",
        "shopeepay",
    ]):
        return (
            "Tidak. Pembayaran SPP dilakukan secara langsung di TPQ "
            "dan dicatat secara manual oleh admin. Informasi mengenai "
            "rekening, transfer, QRIS, atau pembayaran online belum "
            "tersedia dalam data saya."
        )

    # ============================================================
    # 5. SPP - LIHAT STATUS / RIWAYAT
    # ============================================================

    spp_view_intent = (
        "spp" in text
        and contains_any([
            "status",
            "riwayat",
            "sudah dibayar",
            "telah dibayar",
            "yang sudah dibayar",
            "yang telah dibayar",
            "kapan dibayar",
            "kapan terakhir dibayar",
            "terakhir dibayar",
            "tanggal pembayaran",
            "tanggal bayar",
            "bulan yang sudah dibayar",
            "pembayaran yang sudah",
            "data pembayaran",
            "cek pembayaran",
            "cek spp",
            "melihat pembayaran",
            "lihat pembayaran",
            "melihat spp",
            "lihat spp",
            "mengecek spp",
            "mengecek pembayaran",
            "mengetahui pembayaran",
            "mengetahui status",
            "spp sudah",
            "spp belum",
            "lunas",
            "belum lunas",
        ])
    )

    if spp_view_intent:
        if management_action:
            return (
                "Menu SPP pada akun Wali Santri digunakan untuk melihat "
                "status dan riwayat pembayaran. Informasi mengenai "
                "perubahan atau penghapusan data pembayaran belum "
                "tersedia dalam data saya. Silakan hubungi admin TPQ."
            )

        return (
            "Status dan riwayat pembayaran SPP anak dapat dilihat "
            "melalui menu SPP."
        )

    # ============================================================
    # 6. SPP - MELAKUKAN PEMBAYARAN
    # ============================================================

    if "spp" in text and contains_any([
        "bayar",
        "membayar",
        "pembayaran",
        "cara bayar",
        "cara membayar",
        "ingin bayar",
        "ingin membayar",
        "mau bayar",
        "mau membayar",
        "harus bayar",
        "melakukan pembayaran",
        "bayarkan",
    ]):
        return (
            "Pembayaran SPP dilakukan secara langsung di TPQ dan "
            "dicatat secara manual oleh admin."
        )

    # ============================================================
    # 7. MULTI-FEATURE WALI
    # ============================================================

    feature_groups = {
        "absensi": [
            "absensi",
            "absen",
            "kehadiran",
        ],
        "nilai": [
            "nilai",
            "rapor",
            "raport",
        ],
        "hafalan": [
            "hafalan",
        ],
        "prestasi": [
            "prestasi",
        ],
        "tabungan": [
            "tabungan",
        ],
        "spp": [
            "spp",
        ],
        "pengumuman": [
            "pengumuman",
        ],
    }

    mentioned_features = []

    for feature, keywords in feature_groups.items():
        if contains_any(keywords):
            mentioned_features.append(feature)

    # ------------------------------------------------------------
    # Multiple Wali features
    # ------------------------------------------------------------

    if len(mentioned_features) >= 2:

        if management_action:
            return (
                "Akun Wali Santri digunakan untuk melihat informasi "
                "anak, termasuk absensi, nilai dan rapor, hafalan "
                "Juz 30, buku prestasi, tabungan, status SPP, dan "
                "pengumuman. Informasi mengenai penambahan, "
                "perubahan, atau penghapusan data tersebut dari "
                "akun Wali Santri belum tersedia dalam data saya. "
                "Silakan hubungi admin atau ustadz/ustadzah."
            )

        menu_names = []

        if "absensi" in mentioned_features:
            menu_names.append("Absensi")

        if "nilai" in mentioned_features:
            menu_names.append("Nilai & Rapor")

        if "hafalan" in mentioned_features:
            menu_names.append("Hafalan Juz 30")

        if "prestasi" in mentioned_features:
            menu_names.append("Buku Prestasi")

        if "tabungan" in mentioned_features:
            menu_names.append("Tabungan")

        if "spp" in mentioned_features:
            menu_names.append("SPP")

        if "pengumuman" in mentioned_features:
            menu_names.append("Pengumuman")

        return (
            "Wali Santri dapat melihat informasi anak melalui "
            "menu " + ", ".join(menu_names) + "."
        )

    # ============================================================
    # 8. PERKEMBANGAN ANAK
    # ============================================================

    if contains_any([
        "perkembangan anak",
        "perkembangan anak saya",
        "perkembangan santri",
        "memantau perkembangan",
        "memantau perkembangan anak",
        "monitor perkembangan",
        "melihat perkembangan",
        "melihat perkembangan anak",
    ]):
        return (
            "Informasi perkembangan anak dapat dilihat melalui "
            "beberapa menu, seperti Absensi, Nilai & Rapor, "
            "Hafalan Juz 30, dan Buku Prestasi."
        )

    # ============================================================
    # 9. NILAI / RAPOR
    # ============================================================

    if contains_any([
        "nilai",
        "rapor",
        "raport",
    ]):
        if management_action:
            return (
                "Wali Santri dapat melihat nilai dan rapor melalui "
                "menu Nilai & Rapor. Informasi mengenai perubahan, "
                "penambahan, atau penghapusan nilai dari akun Wali "
                "Santri belum tersedia dalam data saya. Silakan "
                "hubungi admin atau ustadz/ustadzah."
            )

        return (
            "Nilai dan rapor anak dapat dilihat melalui menu "
            "Nilai & Rapor."
        )

    # ============================================================
    # 10. ABSENSI
    # ============================================================

    if contains_any([
        "absensi",
        "absen",
        "kehadiran",
    ]):
        if management_action:
            return (
                "Wali Santri dapat melihat riwayat absensi anak "
                "melalui menu Absensi. Informasi mengenai "
                "penambahan, perubahan, atau penghapusan data "
                "absensi dari akun Wali Santri belum tersedia "
                "dalam data saya. Silakan hubungi admin atau "
                "ustadz/ustadzah."
            )

        return (
            "Absensi anak dapat dilihat melalui menu Absensi."
        )

    # ============================================================
    # 11. HAFALAN
    # ============================================================

    if "hafalan" in text:
        if management_action:
            return (
                "Wali Santri dapat melihat hafalan anak melalui "
                "menu Hafalan Juz 30. Informasi mengenai "
                "penambahan, perubahan, atau penghapusan data "
                "hafalan dari akun Wali Santri belum tersedia "
                "dalam data saya. Silakan hubungi admin atau "
                "ustadz/ustadzah."
            )

        return (
            "Hafalan anak dapat dilihat melalui menu Hafalan Juz 30."
        )

    # ============================================================
    # 12. PRESTASI
    # ============================================================

    if "prestasi" in text:
        if management_action:
            return (
                "Wali Santri dapat melihat prestasi anak melalui "
                "menu Buku Prestasi. Informasi mengenai "
                "penambahan, perubahan, atau penghapusan data "
                "prestasi dari akun Wali Santri belum tersedia "
                "dalam data saya. Silakan hubungi admin atau "
                "ustadz/ustadzah."
            )

        return (
            "Prestasi anak dapat dilihat melalui menu Buku Prestasi."
        )

    # ============================================================
    # 13. TABUNGAN
    # ============================================================

    if "tabungan" in text:
        if management_action:
            return (
                "Wali Santri dapat melihat informasi tabungan anak "
                "melalui menu Tabungan. Informasi mengenai "
                "penambahan atau perubahan saldo dari akun Wali "
                "Santri belum tersedia dalam data saya. Silakan "
                "hubungi admin TPQ."
            )

        return (
            "Tabungan anak dapat dilihat melalui menu Tabungan."
        )

    # ============================================================
    # 14. PENGUMUMAN
    # ============================================================

    if "pengumuman" in text:
        if management_action:
            return (
                "Wali Santri dapat membaca pengumuman melalui menu "
                "Pengumuman. Informasi mengenai penambahan, "
                "perubahan, atau penghapusan pengumuman dari akun "
                "Wali Santri belum tersedia dalam data saya. "
                "Silakan hubungi admin TPQ."
            )

        return (
            "Pengumuman dapat dibaca melalui menu Pengumuman."
        )

    # ============================================================
    # 15. FITUR / MENU WALI SANTRI
    # ============================================================

    if contains_any([
        "fitur wali santri",
        "fitur wali",
        "menu wali santri",
        "menu wali",
        "fitur yang dimiliki wali",
        "fitur yang tersedia untuk wali",
        "yang dapat dilakukan wali santri",
        "yang bisa dilakukan wali santri",
        "yang dapat dilakukan oleh wali santri",
        "yang bisa dilakukan oleh wali santri",
        "apa saja yang dapat dilakukan wali",
        "apa saja yang bisa dilakukan wali",
        "wali santri bisa apa",
        "wali santri dapat apa",
        "wali bisa apa",
    ]):
        return (
            "Wali Santri dapat melihat profil santri, absensi, "
            "nilai dan rapor, hafalan Juz 30, buku prestasi, "
            "tabungan, status SPP, dan pengumuman."
        )

    # ============================================================
    # 16. PERTANYAAN UMUM TENTANG AKSES WALI
    # ============================================================

    if contains_any([
        "akses wali",
        "hak akses wali",
        "hak wali",
        "kewenangan wali",
        "wewenang wali",
        "wali boleh apa",
        "wali bisa melakukan apa",
        "wali dapat melakukan apa",
        "apa yang bisa dilakukan wali",
        "apa yang dapat dilakukan wali",
    ]):
        return (
            "Akun Wali Santri digunakan untuk melihat informasi "
            "santri dan data terkait, seperti absensi, nilai dan "
            "rapor, hafalan Juz 30, buku prestasi, tabungan, "
            "status SPP, dan pengumuman."
        )

    # ============================================================
    # 17. DEFAULT
    # ============================================================

    return None


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
    
    # Use deterministic responses for verified Wali Santri navigation.
    verified_response = get_verified_wali_response(user_message)
    if verified_response:
        return verified_response
        
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
