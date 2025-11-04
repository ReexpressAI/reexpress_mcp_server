# Copyright Reexpress AI, Inc. All rights reserved.
"""
Evaluation script for assessing model performance on the document ordering task.
"""

import argparse
import json
import re
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

import sdm_network_constants

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({len(list_to_process)/total})% of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


def extract_final_sentence(response: str, evaluate_first_sentences=False) -> Tuple[str, bool]:
    """
    Extract the final verified sentence from a model response.

    Returns:
        Tuple of (sentence, is_verified)
    """
    # Find all sentence-verification pairs
    sentence_pattern = r'<sentence>(.*?)</sentence>'
    verified_pattern = r'<verified>(.*?)</verified>'

    sentences = re.findall(sentence_pattern, response, re.DOTALL)
    verifications = re.findall(verified_pattern, response, re.DOTALL)

    if evaluate_first_sentences:
        first_sentence = ""
        is_verified = False
        try:
            first_sentence = sentences[0].strip()
            is_verified = verifications[0].strip().lower() == "yes"
        except:
            pass
        return first_sentence, is_verified

    # Find the last verified=Yes sentence, or the last sentence if none are verified
    final_sentence = ""
    is_verified = False

    for i in range(len(sentences)):
        if i < len(verifications):
            if verifications[i].strip().lower() == "yes":
                final_sentence = sentences[i].strip()
                is_verified = True

    # If no verified sentence, take the last one
    if not final_sentence and sentences:
        final_sentence = sentences[-1].strip()
        is_verified = False

    return final_sentence, is_verified


def normalize_sentence(sentence: str) -> str:
    """Normalize a sentence for comparison."""
    # Remove extra whitespace
    sentence = " ".join(sentence.split())
    # preserve case:
    return sentence


def calculate_word_accuracy(predicted: str, reference: str) -> float:
    """Calculate word-level accuracy between two sentences."""
    pred_words = predicted.split()
    ref_words = reference.split()

    if len(pred_words) != len(ref_words):
        return 0.0

    correct = sum(1 for p, r in zip(pred_words, ref_words) if p == r)
    return correct / len(ref_words) if ref_words else 0.0


def evaluate_responses(predictions_file: str, evaluate_first_sentences=False) -> Dict:
    """
    Evaluate model responses against reference responses.

    Args:
        predictions_file: JSONL file with predictions and references

    Returns:
        Dictionary of evaluation metrics
    """

    results = {
        "total": 0,
        "exact_match": 0,
        "verbatim_exact_match_including_tags": 0,
        "verification_tp": 0,
        "verification_fp": 0,
        "verification_fn": 0,
        "verification_tn": 0,
        "no_response": 0,
        "word_accuracies": [],
        "num_attempts": []
    }

    errors = []
    errors_verbatim = []
    errors_unparseable = []

    acc = []
    class_conditional_accuracy = {}
    prediction_conditional_accuracy = {}
    acc_including_unparseable = []
    class_conditional_accuracy_including_unparseable = {}
    prediction_conditional_accuracy_including_unparseable = {}
    for class_i in range(2):
        class_conditional_accuracy[class_i] = []
        prediction_conditional_accuracy[class_i] = []
        class_conditional_accuracy_including_unparseable[class_i] = []
        prediction_conditional_accuracy_including_unparseable[class_i] = []

    with open(predictions_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: {e}")
                continue

            results["total"] += 1

            # Get generated and reference responses
            generated = data.get(sdm_network_constants.GENERATED_RESPONSE_KEY, "")
            multiset = data.get(sdm_network_constants.MULTISET_KEY, "")
            ref_sentence = data.get(sdm_network_constants.ORIGINAL_DOCUMENT_ORDER_KEY, "")
            ref_sentence_with_tags = data.get(sdm_network_constants.REEXPRESS_GENAI_DOCUMENT_KEY, "")

            # Extract sentences from responses
            gen_sentence, gen_verified = extract_final_sentence(generated,
                                                                evaluate_first_sentences=evaluate_first_sentences)
            results["verbatim_exact_match_including_tags"] += \
                int(normalize_sentence(generated) == normalize_sentence(ref_sentence_with_tags))

            if not gen_sentence:
                results["no_response"] += 1
                errors_unparseable.append({
                    "multiset": multiset,
                    "error": "No sentence extracted",
                    "generated_parsed": gen_sentence,
                    "reference_content": ref_sentence,
                    "generated": generated,
                    "reference": ref_sentence_with_tags
                })
                # These are always treated as incorrect predictions:
                acc_including_unparseable.append(0)
                class_conditional_accuracy_including_unparseable[1].append(0)
                prediction_conditional_accuracy_including_unparseable[0].append(0)
                continue

            # Count attempts (number of sentence tags in response)
            num_attempts = len(re.findall(r'<sentence>', generated))
            results["num_attempts"].append(num_attempts)

            # Given the generation, assess the verification classification. E.g., if the model produces an
            # incorrect generation, ideally it will also indicate that the output is not verified (and vice-versa).
            groundtruth_label_approximation = int(normalize_sentence(gen_sentence) == normalize_sentence(ref_sentence))
            verification_prediction = int(gen_verified)
            class_conditional_accuracy[groundtruth_label_approximation].append(
                int(groundtruth_label_approximation == verification_prediction))
            prediction_conditional_accuracy[verification_prediction].append(
                int(groundtruth_label_approximation == verification_prediction))
            acc.append(
                int(groundtruth_label_approximation == verification_prediction))

            class_conditional_accuracy_including_unparseable[groundtruth_label_approximation].append(
                int(groundtruth_label_approximation == verification_prediction))
            prediction_conditional_accuracy_including_unparseable[verification_prediction].append(
                int(groundtruth_label_approximation == verification_prediction))
            acc_including_unparseable.append(
                int(groundtruth_label_approximation == verification_prediction))

            # Check exact match of content
            if normalize_sentence(gen_sentence) == normalize_sentence(ref_sentence):
                results["exact_match"] += 1

                if gen_verified:
                    results["verification_tp"] += 1
                else:
                    results["verification_fn"] += 1
                    errors.append({
                        "multiset": multiset,
                        "error": "Verification False Negative (tag is No, but content is correct)",
                        "generated_parsed": gen_sentence,
                        "reference_content": ref_sentence,
                        "generated": generated,
                        "reference": ref_sentence_with_tags
                    })
            else:
                if gen_verified:
                    results["verification_fp"] += 1
                    errors.append({
                        "multiset": multiset,
                        "error": "Verification False Positive (tag is Yes, but content is incorrect)",
                        "generated_parsed": gen_sentence,
                        "reference_content": ref_sentence,
                        "generated": generated,
                        "reference": ref_sentence_with_tags
                    })
                else:
                    results["verification_tn"] += 1
            # also collect any instances that are otherwise correct (content and tags), but with extraneous text
            if int(normalize_sentence(generated) != normalize_sentence(ref_sentence_with_tags)) \
                    and normalize_sentence(gen_sentence) == normalize_sentence(ref_sentence) and gen_verified:
                errors_verbatim.append({
                    "multiset": multiset,
                    "error": "Generation contains extraneous content",
                    "generated_parsed": gen_sentence,
                    "reference_content": ref_sentence,
                    "generated": generated,
                    "reference": ref_sentence_with_tags
                })

            # Calculate word-level accuracy
            word_acc = calculate_word_accuracy(
                normalize_sentence(gen_sentence),
                normalize_sentence(ref_sentence)
            )
            results["word_accuracies"].append(word_acc)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS -- Classification using the verification tags")
    print("=" * 60)
    print_summary(f"Mean Verification Acc., among parseable outputs", acc, total=len(acc))
    for class_i in range(2):
        print_summary(f"Class-conditional Verification Acc. (label: {class_i}), among parseable outputs",
                      class_conditional_accuracy[class_i], total=len(acc))
        print_summary(f"Prediction-conditional Verification Acc. (label: {class_i}), among parseable outputs",
                      prediction_conditional_accuracy[class_i], total=len(acc))
    print("")
    assert len(acc_including_unparseable) == results["total"]
    print_summary(f"Mean Verification Acc., among all outputs, "
                  f"with unparseable set to true class 1 and predicted class 0", acc_including_unparseable,
                  total=results["total"])
    for class_i in range(2):
        print_summary(f"Class-conditional Verification Acc. (label: {class_i}), among all outputs",
                      class_conditional_accuracy_including_unparseable[class_i], total=results["total"])
        print_summary(f"Prediction-conditional Verification Acc. (label: {class_i}), among all outputs",
                      prediction_conditional_accuracy_including_unparseable[class_i], total=results["total"])
    # Compute aggregate metrics
    if results["total"] > 0:
        results["exact_match_rate"] = results["exact_match"] / results["total"]
        results["verbatim_exact_match_including_tags_rate"] = \
            results["verbatim_exact_match_including_tags"] / results["total"]
        results["verification_tp_accuracy"] = results["verification_tp"] / results["total"]
        results["verification_tn_accuracy"] = results["verification_tn"] / results["total"]

        if results["word_accuracies"]:
            results["mean_word_accuracy"] = np.mean(results["word_accuracies"])
            results["std_word_accuracy"] = np.std(results["word_accuracies"])

        if results["num_attempts"]:
            results["mean_attempts"] = np.mean(results["num_attempts"])
            results["max_attempts"] = max(results["num_attempts"])

    # Store errors for analysis
    results["errors"] = errors
    results["errors_verbatim"] = errors_verbatim
    results["errors_unparseable"] = errors_unparseable

    return results


def compare_models(baseline_file: str, finetuned_file: str) -> Dict:
    """Compare performance between baseline and fine-tuned models."""

    logger.info("Evaluating baseline model...")
    baseline_metrics = evaluate_responses(baseline_file)

    logger.info("Evaluating fine-tuned model...")
    finetuned_metrics = evaluate_responses(finetuned_file)

    comparison = {
        "baseline": {
            "verbatim_exact_match_including_tags_rate":
                baseline_metrics.get("verbatim_exact_match_including_tags_rate", 0),
            "exact_match_rate": baseline_metrics.get("exact_match_rate", 0),
            "verification_tp_accuracy": baseline_metrics.get("verification_tp_accuracy", 0),
            "verification_tn_accuracy": baseline_metrics.get("verification_tn_accuracy", 0),
            "mean_word_accuracy": baseline_metrics.get("mean_word_accuracy", 0),
            "mean_attempts": baseline_metrics.get("mean_attempts", 0)
        },
        "finetuned": {
            "verbatim_exact_match_including_tags_rate":
                finetuned_metrics.get("verbatim_exact_match_including_tags_rate", 0),
            "exact_match_rate": finetuned_metrics.get("exact_match_rate", 0),
            "verification_tp_accuracy": finetuned_metrics.get("verification_tp_accuracy", 0),
            "verification_tn_accuracy": finetuned_metrics.get("verification_tn_accuracy", 0),
            "mean_word_accuracy": finetuned_metrics.get("mean_word_accuracy", 0),
            "mean_attempts": finetuned_metrics.get("mean_attempts", 0)
        },
        "improvements": {}
    }

    # Calculate improvements
    for metric in comparison["baseline"]:
        baseline_val = comparison["baseline"][metric]
        finetuned_val = comparison["finetuned"][metric]

        if baseline_val > 0:
            improvement = ((finetuned_val - baseline_val) / baseline_val) * 100
            comparison["improvements"][metric] = f"{improvement:+.2f}%"

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance on document ordering task")

    # Input files
    parser.add_argument("--predictions_file", type=str, required=True,
                        help="JSONL file with predictions and references")
    parser.add_argument("--baseline_file", type=str,
                        help="Baseline predictions for comparison")

    # Output
    parser.add_argument("--output_file", type=str,
                        help="Output file for metrics (JSON)")
    parser.add_argument("--errors_file", type=str,
                        help="Output file for error analysis")
    parser.add_argument("--verbatim_errors_file", type=str,
                        help="Output file for error analysis")
    parser.add_argument("--unparseable_errors_file", type=str,
                        help="Output file for error analysis")

    # Options
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")
    parser.add_argument("--evaluate_first_sentences", action="store_true", default=False,
                        help="Evaluate the first generated full sentence rather than the last.")

    args = parser.parse_args()

    if args.baseline_file:
        # Compare two models
        logger.info("Comparing baseline and fine-tuned models...")
        results = compare_models(args.baseline_file, args.predictions_file)

        # Print comparison
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        print("\nBaseline Model:")
        for metric, value in results["baseline"].items():
            print(f"  {metric}: {value:.4f}")

        print("\nFine-tuned Model:")
        for metric, value in results["finetuned"].items():
            print(f"  {metric}: {value:.4f}")

        print("\nImprovements:")
        for metric, value in results["improvements"].items():
            print(f"  {metric}: {value}")

    else:
        # Evaluate single model
        logger.info(f"Evaluating {args.predictions_file}")
        results = evaluate_responses(
            args.predictions_file,
            evaluate_first_sentences=args.evaluate_first_sentences
        )

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nTotal examples: {results['total']}")
        print(f"Verbatim exact matches (including formatting): "
              f"{results['verbatim_exact_match_including_tags']} "
              f"({results.get('verbatim_exact_match_including_tags_rate', 0):.2%})")
        print(f"Exact matches (content only, {'first' if args.evaluate_first_sentences else 'last'} "
              f"available tagged sentence): {results['exact_match']} ({results.get('exact_match_rate', 0):.2%})")
        print(f"No response: {results['no_response']} ({results['no_response'] / results['total']:.2%})")
        print("")
        print("Additional reference counts among parseable:")
        print(f"\tverification_tp: {results['verification_tp']} ({results['verification_tp'] / results['total']:.2%})")
        print(
            f"\tverification_fp: {results['verification_fp']} "
            f"({results['verification_fp'] / results['total']:.2%})")
        print(f"\tverification_fn: {results['verification_fn']} ({results['verification_fn'] / results['total']:.2%})")
        print(f"\tverification_tn: {results['verification_tn']} ({results['verification_tn'] / results['total']:.2%})")

        if "mean_word_accuracy" in results:
            print(f"\nWord-level accuracy: {results['mean_word_accuracy']:.4f} ± {results['std_word_accuracy']:.4f}")

        if "mean_attempts" in results:
            print(f"Average attempts: {results['mean_attempts']:.2f} (max: {results['max_attempts']})")

        count_errors_to_print = 10

        if args.verbose and results.get("errors_unparseable"):
            print("\n" + "=" * 60)
            print(f"ERROR ANALYSIS (first {count_errors_to_print}), "
                  f"among generations that cannot be parsed.")
            print("=" * 60)

            for i, error in enumerate(results["errors_unparseable"][0:count_errors_to_print]):
                print(f"\n{i}. Multiset: {error['multiset']}")
                print(f"   Error: {error['error']}")
                if 'generated' in error:
                    print(f"   Generated (full): {error['generated']}")
                if 'reference' in error:
                    print(f"   Reference (with tags): {error['reference']}")

        if args.verbose and results.get("errors"):
            print("\n" + "=" * 60)
            print(f"ERROR ANALYSIS (first {count_errors_to_print}), "
                  f"among parseable generations with incorrect content.")
            print("=" * 60)

            for i, error in enumerate(results["errors"][0:count_errors_to_print]):
                print(f"\n{i}. Multiset: {error['multiset']}")
                print(f"   Error: {error['error']}")
                if 'generated_parsed' in error:
                    print(f"   Generated (parsed): {error['generated_parsed']}")
                if 'reference_content' in error:
                    print(f"   Reference (content): {error['reference_content']}")

        if args.verbose and results.get("errors_verbatim"):
            print("\n" + "=" * 60)
            print(f"ERROR ANALYSIS (first {count_errors_to_print}), "
                  f"among generations that are parseable with correct content, "
                  f"but extraneous information is included.")
            print("=" * 60)

            for i, error in enumerate(results["errors_verbatim"][0:count_errors_to_print]):
                print(f"\n{i}. Multiset: {error['multiset']}")
                print(f"   Error: {error['error']}")
                if 'generated' in error:
                    print(f"   Generated (full): {error['generated']}")
                if 'reference' in error:
                    print(f"   Reference (with tags): {error['reference']}")

    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Metrics saved to {args.output_file}")

    # Save errors for detailed analysis
    if args.errors_file and results.get("errors"):
        with open(args.errors_file, 'w') as f:
            for error in results["errors"]:
                f.write(json.dumps(error) + "\n")
        print(f"Errors saved to {args.errors_file}")

    if args.verbatim_errors_file and results.get("errors_verbatim"):
        with open(args.verbatim_errors_file, 'w') as f:
            for error in results["errors_verbatim"]:
                f.write(json.dumps(error) + "\n")
        print(f"Errors saved to {args.verbatim_errors_file}")

    if args.unparseable_errors_file and results.get("errors_unparseable"):
        with open(args.unparseable_errors_file, 'w') as f:
            for error in results["errors_unparseable"]:
                f.write(json.dumps(error) + "\n")
        print(f"Errors saved to {args.unparseable_errors_file}")


if __name__ == "__main__":
    main()
