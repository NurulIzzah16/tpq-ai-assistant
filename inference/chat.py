"""
TPQ AI Assistant - Interactive CLI Chat

Provides a command-line interface for testing model inference.

Usage:
    python inference/chat.py

Type 'quit' or 'exit' to end the conversation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.model_loader import load_model, generate_response


def main():
    print("=" * 60)
    print("  TPQ AI Assistant - Interactive Chat")
    print("=" * 60)
    print()
    print("Loading model... (this may take a moment)")
    print()

    # Load model once
    model, tokenizer = load_model()

    print()
    print("-" * 60)
    print("Model loaded! You can start chatting.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 60)
    print()

    while True:
        try:
            # Get user input
            user_input = input("Anda: ").strip()

            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "q", "keluar"):
                print("\nTerima kasih telah menggunakan TPQ AI Assistant!")
                break

            # Skip empty input
            if not user_input:
                print("(Silakan ketik pertanyaan Anda)")
                continue

            # Generate response
            print("\nTPQ AI: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nTerima kasih telah menggunakan TPQ AI Assistant!")
            break
        except Exception as e:
            print(f"\n[Error] Terjadi kesalahan: {e}")
            print("Silakan coba lagi.\n")


if __name__ == "__main__":
    main()
