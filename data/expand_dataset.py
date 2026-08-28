"""
TPQ AI Assistant - Dataset Expansion Helper

This script provides templates and utilities to expand the dataset
from 100 to 300+ examples. It generates variation templates that
can be reviewed and customized before adding to the dataset.

Usage:
    python data/expand_dataset.py
"""

import json
import os
import random

# Template categories with question patterns and answer templates
EXPANSION_TEMPLATES = {
    "informasi_umum": {
        "questions": [
            "Apa visi dan misi TPQ?",
            "Berapa lama program pendidikan di TPQ?",
            "Apakah TPQ terakreditasi?",
            "Apa perbedaan TPQ dan TPA?",
            "Sejak kapan TPQ ini berdiri?",
            "Apa motto TPQ?",
            "Bagaimana sejarah berdirinya TPQ?",
            "Siapa pendiri TPQ?",
            "Apa tujuan utama pendidikan di TPQ?",
            "Berapa jumlah santri yang terdaftar saat ini?",
        ],
        "answer_hints": [
            "Jelaskan visi misi TPQ yang relevan dengan pendidikan Al-Quran",
            "Durasi program bervariasi, bisa 2-4 tahun tergantung kemampuan santri",
            "Status akreditasi tergantung kebijakan lembaga",
            "TPQ dan TPA memiliki fokus serupa, perbedaannya pada cakupan kurikulum",
            "Tahun berdiri sesuai data lembaga",
            "Motto TPQ sesuai kebijakan lembaga",
            "Sejarah berdirinya sesuai data lembaga",
            "Pendiri sesuai data lembaga",
            "Tujuan: membentuk generasi Qurani yang berakhlak mulia",
            "Jumlah santri sesuai data terkini",
        ],
    },
    "absensi": {
        "questions": [
            "Apakah absensi dilakukan secara digital?",
            "Siapa yang mengisi absensi santri?",
            "Bagaimana jika absensi anak saya salah?",
            "Apakah ada rekap absensi tahunan?",
            "Bagaimana cara mengajukan izin tidak hadir?",
            "Apakah wali santri mendapat notifikasi jika anak tidak hadir?",
            "Apakah absensi mempengaruhi kenaikan kelas?",
            "Berapa persen kehadiran minimum untuk naik kelas?",
            "Bagaimana pencatatan absensi saat kegiatan tambahan?",
            "Apakah ada sistem absensi dengan sidik jari?",
        ],
        "answer_hints": [
            "Absensi dilakukan secara digital melalui sistem administrasi",
            "Ustadz/ustadzah pengampu yang mengisi absensi setiap pertemuan",
            "Hubungi admin untuk koreksi data absensi",
            "Rekap absensi tahunan tersedia di menu Laporan",
            "Izin diajukan melalui sistem atau menghubungi ustadz/ustadzah",
            "Notifikasi otomatis jika santri alfa",
            "Kehadiran menjadi salah satu syarat kenaikan kelas",
            "Minimal 75% kehadiran untuk naik kelas",
            "Absensi kegiatan tambahan dicatat terpisah",
            "Sistem absensi menggunakan pencatatan digital di sistem administrasi",
        ],
    },
    "nilai": {
        "questions": [
            "Apakah ada remedial untuk santri yang nilainya kurang?",
            "Bagaimana cara mengetahui target hafalan semester ini?",
            "Apakah nilai bisa diakses oleh santri sendiri?",
            "Bagaimana format rapor digital di TPQ?",
            "Apakah ada ranking atau peringkat kelas?",
            "Bagaimana cara membaca skala penilaian di rapor?",
            "Apakah ada penilaian sikap dan akhlak?",
            "Bagaimana jika nilai ujian anak saya rendah?",
            "Apakah ada ujian lisan selain ujian tertulis?",
            "Bagaimana penilaian untuk kelas Iqra?",
        ],
        "answer_hints": [
            "Program remedial tersedia untuk santri yang belum mencapai nilai minimum",
            "Target hafalan tersedia di kurikulum dan dapat ditanyakan ke ustadz/ustadzah",
            "Nilai diakses melalui akun wali santri",
            "Rapor digital berisi rangkuman nilai semua komponen per semester",
            "Kebijakan ranking tergantung masing-masing TPQ",
            "Skala: A (86-100), B (71-85), C (56-70), D (<56)",
            "Penilaian sikap dan akhlak tercantum di rapor sebagai catatan deskriptif",
            "Konsultasi dengan ustadz/ustadzah untuk bimbingan tambahan",
            "Ujian di TPQ umumnya berupa ujian lisan (membaca Al-Quran dan hafalan)",
            "Penilaian kelas Iqra berdasarkan kelancaran dan ketepatan membaca setiap jilid",
        ],
    },
    "pembayaran": {
        "questions": [
            "Apakah ada biaya kegiatan ekstra di luar SPP?",
            "Bagaimana cara mendapatkan kuitansi pembayaran?",
            "Apakah pembayaran bisa dilakukan secara online?",
            "Apa saja metode pembayaran yang diterima?",
            "Bagaimana jika terjadi kesalahan dalam pencatatan pembayaran?",
            "Apakah ada diskon SPP untuk saudara kandung?",
            "Berapa biaya buku Iqra dan Al-Quran?",
            "Apakah ada iuran untuk kegiatan akhir tahun?",
            "Bagaimana cara mengajukan cicilan pembayaran?",
            "Apakah ada biaya seragam?",
        ],
        "answer_hints": [
            "Biaya kegiatan ekstra diinformasikan melalui pengumuman terpisah",
            "Kuitansi tersedia di menu Pembayaran atau dari administrasi langsung",
            "Pembayaran online melalui transfer bank ke rekening resmi TPQ",
            "Transfer bank, tunai di kantor administrasi",
            "Hubungi admin untuk koreksi data pembayaran",
            "Kebijakan diskon untuk saudara kandung tergantung aturan TPQ",
            "Biaya buku terpisah dari SPP, informasi harga di administrasi",
            "Iuran kegiatan akhir tahun diinformasikan melalui pengumuman",
            "Pengajuan cicilan dapat dibicarakan dengan administrasi",
            "Biaya seragam terpisah dari SPP jika TPQ mewajibkan seragam",
        ],
    },
    "out_of_domain": {
        "questions": [
            "Bagaimana cara investasi saham?",
            "Apa rumus matematika untuk luas lingkaran?",
            "Tolong terjemahkan ke bahasa Inggris",
            "Siapa pemenang Piala Dunia terakhir?",
            "Bagaimana cara membuat website?",
            "Apa obat untuk sakit kepala?",
            "Bagaimana cuaca hari ini?",
            "Tolong buatkan surat lamaran kerja",
            "Apa rekomendasi film terbaik?",
            "Bagaimana cara diet yang sehat?",
        ],
        "answer_hints": [
            "Maaf, pertanyaan di luar cakupan. Saya khusus membantu administrasi TPQ.",
        ] * 10,
    },
}


def generate_expansion_examples():
    """Generate expansion examples from templates."""
    examples = []

    for category, data in EXPANSION_TEMPLATES.items():
        for question, hint in zip(data["questions"], data["answer_hints"]):
            if category == "out_of_domain":
                answer = (
                    f"Maaf, pertanyaan tersebut di luar cakupan layanan saya. "
                    f"Saya adalah TPQ AI Assistant yang khusus membantu pertanyaan "
                    f"terkait administrasi Taman Pendidikan Al-Quran (TPQ). "
                    f"Silakan ajukan pertanyaan seputar TPQ seperti pendaftaran, "
                    f"absensi, nilai, pembayaran SPP, atau jadwal belajar."
                )
            else:
                answer = f"[TEMPLATE - PERLU DIEDIT] {hint}"

            example = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            examples.append(example)

    return examples


def save_expansion_draft(examples, output_path):
    """Save expansion draft to JSONL file for review."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} expansion examples to: {output_path}")


def merge_datasets(original_path, expansion_path, output_path):
    """Merge original and expansion datasets."""
    examples = []

    # Load original
    with open(original_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    original_count = len(examples)

    # Load expansion
    with open(expansion_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                example = json.loads(line)
                # Skip template placeholders that haven't been edited
                assistant_msg = example["messages"][1]["content"]
                if not assistant_msg.startswith("[TEMPLATE"):
                    examples.append(example)
    expansion_count = len(examples) - original_count

    # Save merged
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nMerge completed:")
    print(f"  Original examples: {original_count}")
    print(f"  Expansion examples: {expansion_count}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Saved to: {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw")
    original_path = os.path.join(raw_dir, "dataset.jsonl")
    expansion_path = os.path.join(raw_dir, "dataset_expansion_draft.jsonl")

    print("=" * 60)
    print("TPQ AI Assistant - Dataset Expansion Helper")
    print("=" * 60)

    # Generate expansion draft
    examples = generate_expansion_examples()
    save_expansion_draft(examples, expansion_path)

    print(f"\nCategories generated:")
    for category, data in EXPANSION_TEMPLATES.items():
        print(f"  - {category}: {len(data['questions'])} examples")

    print(f"\nTotal expansion examples: {len(examples)}")
    print(f"\nNext steps:")
    print(f"  1. Open: {expansion_path}")
    print(f"  2. Edit [TEMPLATE] answers with proper responses")
    print(f"  3. Out-of-domain examples are already complete")
    print(f"  4. Run merge:")
    print(f"     python data/expand_dataset.py --merge")

    # Check if merge mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        if os.path.exists(expansion_path):
            merge_datasets(original_path, expansion_path, original_path)
        else:
            print(f"\nError: Expansion file not found: {expansion_path}")
            print("Run without --merge first to generate the expansion draft.")


if __name__ == "__main__":
    main()
