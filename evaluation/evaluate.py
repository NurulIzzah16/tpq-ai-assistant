"""
TPQ AI Assistant - Model Evaluation

Evaluates and compares the base Qwen model vs the fine-tuned model
on test questions. Supports both automated keyword scoring and
manual scoring workflow.

Usage:
    # Full evaluation (requires 2 models loaded)
    python evaluation/evaluate.py

    # Generate manual scoring template only
    python evaluation/evaluate.py --manual-only

    # Evaluate fine-tuned model only
    python evaluation/evaluate.py --finetuned-only
"""

import json
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_test_questions(filepath):
    """Load test questions from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def score_response_auto(response, expected_keywords, is_in_domain):
    """
    Automatically score a response based on keyword matching.

    Scoring (1-5):
        5: All keywords present, well-formed response
        4: Most keywords present
        3: Some keywords present
        2: Few keywords present
        1: No keywords or irrelevant response

    For out-of-domain questions, checks if the model correctly
    identifies the question as outside its scope.
    """
    response_lower = response.lower()

    if not response.strip():
        return 1

    matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    total = len(expected_keywords)

    if total == 0:
        return 3  # No keywords to check

    ratio = matched / total

    if ratio >= 0.8:
        return 5
    elif ratio >= 0.6:
        return 4
    elif ratio >= 0.4:
        return 3
    elif ratio >= 0.2:
        return 2
    else:
        return 1


def evaluate_model(model, tokenizer, questions, model_label="model"):
    """
    Run evaluation on a set of test questions.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        questions: List of test question dicts.
        model_label: Label for this model (e.g., 'base' or 'finetuned').

    Returns:
        list: Results with questions, responses, and scores.
    """
    from inference.model_loader import generate_response

    results = []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:50]}...")

        try:
            response = generate_response(model, tokenizer, q["question"])
        except Exception as e:
            response = f"[ERROR] {str(e)}"

        auto_score = score_response_auto(
            response, q["expected_keywords"], q["is_in_domain"]
        )

        result = {
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "is_in_domain": q["is_in_domain"],
            f"{model_label}_response": response,
            f"{model_label}_auto_score": auto_score,
            f"{model_label}_manual_score": None,  # To be filled manually
        }
        results.append(result)

    return results


def generate_manual_template(questions, output_path):
    """
    Generate a manual scoring template JSON file.

    This template can be filled in by a human evaluator who reads
    the responses and assigns scores from 1-5.
    """
    template = {
        "evaluation_date": datetime.now().isoformat(),
        "scoring_guide": {
            "5": "Excellent - Accurate, complete, well-formatted, in correct domain",
            "4": "Good - Mostly accurate, minor issues",
            "3": "Average - Partially correct, missing some information",
            "2": "Poor - Mostly incorrect or irrelevant",
            "1": "Very Poor - Completely wrong or no response",
        },
        "criteria": [
            "Relevance: Is the response relevant to the question?",
            "Correctness: Is the information accurate?",
            "Instruction Following: Does the model follow its role as TPQ assistant?",
            "Response Quality: Is the response clear, polite, and well-formed?",
        ],
        "questions": [],
    }

    for q in questions:
        template["questions"].append(
            {
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "is_in_domain": q["is_in_domain"],
                "base_model_response": "[Run evaluation first]",
                "base_model_score": None,
                "finetuned_model_response": "[Run evaluation first]",
                "finetuned_model_score": None,
                "notes": "",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"Manual scoring template saved to: {output_path}")


def print_summary(results, model_label):
    """Print evaluation summary statistics."""
    scores = [r[f"{model_label}_auto_score"] for r in results if r[f"{model_label}_auto_score"] is not None]

    if not scores:
        print(f"  No scores available for {model_label}")
        return

    avg = sum(scores) / len(scores)
    in_domain = [
        r[f"{model_label}_auto_score"]
        for r in results
        if r["is_in_domain"] and r[f"{model_label}_auto_score"] is not None
    ]
    out_domain = [
        r[f"{model_label}_auto_score"]
        for r in results
        if not r["is_in_domain"] and r[f"{model_label}_auto_score"] is not None
    ]

    print(f"\n  {model_label.upper()} Model Summary:")
    print(f"    Overall avg score: {avg:.2f} / 5.00")
    if in_domain:
        print(f"    In-domain avg:     {sum(in_domain)/len(in_domain):.2f} / 5.00")
    if out_domain:
        print(f"    Out-of-domain avg: {sum(out_domain)/len(out_domain):.2f} / 5.00")
    print(f"    Total questions:   {len(scores)}")


def main():
    parser = argparse.ArgumentParser(description="TPQ AI Assistant - Evaluation")
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Only generate manual scoring template without running model inference",
    )
    parser.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Only evaluate the fine-tuned model (skip base model)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TPQ AI Assistant - Model Evaluation")
    print("=" * 60)
    print()

    # Paths
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(eval_dir, "test_questions.json")
    results_path = os.path.join(eval_dir, "results.json")
    manual_template_path = os.path.join(eval_dir, "manual_scoring_template.json")

    # Load test questions
    print("[1/4] Loading test questions...")
    questions = load_test_questions(questions_path)
    print(f"  Loaded {len(questions)} questions")
    print()

    # Generate manual template (always)
    print("[2/4] Generating manual scoring template...")
    generate_manual_template(questions, manual_template_path)
    print()

    if args.manual_only:
        print("Manual-only mode. Skipping model inference.")
        print(f"Fill in scores in: {manual_template_path}")
        return

    from inference.model_loader import load_model
    from training.config import MODEL_NAME, OUTPUT_DIR

    all_results = []

    # Evaluate base model
    if not args.finetuned_only:
        print("[3/4] Evaluating BASE model...")
        print(f"  Loading base model: {MODEL_NAME}")
        try:
            base_model, base_tokenizer = load_model(
                model_path="__nonexistent__",  # Force base model fallback
                model_name=MODEL_NAME,
            )
            base_results = evaluate_model(
                base_model, base_tokenizer, questions, "base"
            )
            print_summary(base_results, "base")
            all_results = base_results

            # Free memory
            del base_model, base_tokenizer
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] Could not load base model: {e}")
            print("  Skipping base model evaluation.")
            all_results = [{"id": q["id"], "question": q["question"],
                           "category": q["category"], "is_in_domain": q["is_in_domain"]}
                          for q in questions]
    else:
        all_results = [{"id": q["id"], "question": q["question"],
                       "category": q["category"], "is_in_domain": q["is_in_domain"]}
                      for q in questions]

    # Evaluate fine-tuned model
    print()
    print("[4/4] Evaluating FINE-TUNED model...")
    print(f"  Loading fine-tuned model from: {OUTPUT_DIR}")

    try:
        ft_model, ft_tokenizer = load_model(model_path=OUTPUT_DIR)
        ft_results = evaluate_model(ft_model, ft_tokenizer, questions, "finetuned")

        # Merge results
        for i, ft_result in enumerate(ft_results):
            for key, value in ft_result.items():
                if key.startswith("finetuned"):
                    all_results[i][key] = value

        print_summary(ft_results, "finetuned")

    except Exception as e:
        print(f"  [ERROR] Could not load fine-tuned model: {e}")
        print("  Skipping fine-tuned model evaluation.")
        print("  Make sure you have trained the model first: python training/train.py")

    # Save results
    print()
    print("-" * 60)
    output = {
        "evaluation_date": datetime.now().isoformat(),
        "num_questions": len(questions),
        "results": all_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {results_path}")
    print(f"Manual template: {manual_template_path}")
    print()
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
