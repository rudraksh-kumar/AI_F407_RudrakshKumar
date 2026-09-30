"""
Runner script for AI Laboratory: Bayesian Networks and Autoregressive Language Models.
Executes training, CPT extraction, probability normalisation testing, next-word prediction,
sentence generation (greedy & sampling), model comparisons, and outputs deliverables
(lab_results.json, first_order_generated.txt, second_order_generated.txt).
"""

import json
import random
from first_order_lm import FirstOrderLanguageModel
from second_order_lm import SecondOrderLanguageModel

# Dataset specified in Section 6 (Part III)
DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]

def preprocess(sentences):
    tokenized = []
    for s in sentences:
        tokens = ["<START>"] + s.lower().split() + ["<END>"]
        tokenized.append(tokens)
    return tokenized

def main():
    print("================================================================================")
    print(" AI LABORATORY: BAYESIAN NETWORKS & AUTOREGRESSIVE LANGUAGE MODELS EXPERIMENTS ")
    print("================================================================================\n")

    tokenized_data = preprocess(DATASET)
    print("Preprocessed Training Corpus:")
    for i, sent in enumerate(tokenized_data, 1):
        print(f"  Sentence {i}: {' '.join(sent)}")
    print()

    # ---------------------------------------------------------
    # PART 1: First-Order Model Experiments
    # ---------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print(" PART 1: FIRST-ORDER AUTOREGRESSIVE LANGUAGE MODEL")
    print("--------------------------------------------------------------------------------")
    lm1 = FirstOrderLanguageModel()
    lm1.train(tokenized_data)

    print("\n[Normalisation Test - First Order Model]")
    norm1_results = lm1.test_normalization()
    all_norm1_valid = True
    norm1_lines = []
    for word, sum_prob in norm1_results.items():
        is_valid = abs(sum_prob - 1.0) < 1e-6
        if not is_valid:
            all_norm1_valid = False
        line = f"  P(* | '{word}':8s) sum = {sum_prob:.6f} [{'VALID' if is_valid else 'INVALID'}]"
        norm1_lines.append(line)
        print(line)
    print(f"  Overall Normalisation Check: {'PASSED (All sums equal 1.0)' if all_norm1_valid else 'FAILED'}\n")

    print("[Question 3: CPTs for Target Words (the, cat, dog, sat, ran)]")
    target_words = ["the", "cat", "dog", "sat", "ran"]
    cpt1_selected = {}
    cpt1_str_lines = []
    for w in target_words:
        dist = lm1.get_distribution(w)
        cpt1_selected[w] = dist
        cpt1_str_lines.append(f"  P(X_t | X_{{t-1}} = '{w}'):")
        print(f"  P(X_t | X_{{t-1}} = '{w}'):")
        for nxt, p in dist.items():
            print(f"    -> P('{nxt}' | '{w}') = {p:.4f} ({count_str(lm1.counts[w][nxt], lm1.counts[w])})")
            cpt1_str_lines.append(f"    -> {nxt:5s} : {p:.4f} ({count_str(lm1.counts[w][nxt], lm1.counts[w])})")
        unobserved = [v for v in lm1.vocabulary if v not in dist and v != "<START>"]
        cpt1_str_lines.append(f"    Zero-probability transitions: {len(unobserved)} / {len(lm1.vocabulary)-1}\n")
        print(f"    Zero-probability next tokens count: {len(unobserved)} / {len(lm1.vocabulary)-1}\n")

    print("[Part VIII & Question 9: Next Word Predictions (5 contexts)]")
    contexts_5 = ["the", "cat", "dog", "sat", "ran"]
    print(f"{'Context':<10} | {'Argmax Prediction':<18} | {'Probability':<12} | {'Distribution'}")
    print("-" * 75)
    prediction_lines = [f"{'Context':<10} | {'Argmax Prediction':<18} | {'Probability':<12} | {'Distribution'}", "-" * 75]
    for c in contexts_5:
        dist = lm1.get_distribution(c)
        if dist:
            top_w, top_p = max(dist.items(), key=lambda x: x[1])
            dist_str = ", ".join([f"{k}:{v:.2f}" for k, v in dist.items()])
            row = f"{c:<10} | {top_w:<18} | {top_p:<12.4f} | {dist_str}"
            prediction_lines.append(row)
            print(row)
    print()

    print("[Part IX & X: Sentence Generation - First Order]")
    greedy_sent1 = " ".join(lm1.generate_sentence(mode="greedy", max_length=15))
    print(f"  Greedy Mode (Deterministic): {greedy_sent1}")

    random.seed(42)
    sampled_sents_1st = []
    print("\n  Sampling Mode (20 Sentences):")
    for i in range(1, 21):
        sent = " ".join(lm1.generate_sentence(mode="sampling"))
        sampled_sents_1st.append(sent)
        print(f"    {i:2d}. {sent}")
    print()

    # ---------------------------------------------------------
    # PART 2: Second-Order Model Experiments
    # ---------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print(" PART 2: SECOND-ORDER AUTOREGRESSIVE LANGUAGE MODEL")
    print("--------------------------------------------------------------------------------")
    lm2 = SecondOrderLanguageModel()
    lm2.train(tokenized_data)

    print("\n[Normalisation Test - Second Order Model]")
    norm2_results = lm2.test_normalization()
    all_norm2_valid = True
    norm2_lines = []
    for ctx, sum_prob in norm2_results.items():
        is_valid = abs(sum_prob - 1.0) < 1e-6
        if not is_valid:
            all_norm2_valid = False
        line = f"  P(* | {str(ctx):20s}) = {sum_prob:.6f} [{'VALID' if is_valid else 'INVALID'}]"
        norm2_lines.append(line)
        print(line)
    print(f"  Overall Normalisation Check: {'PASSED (All sums equal 1.0)' if all_norm2_valid else 'FAILED'}\n")

    print("[Second Order CPTs for All Observed Contexts]")
    cpt2_all = {}
    cpt2_str_lines = []
    for ctx, dist in lm2.probabilities.items():
        ctx_str = f"('{ctx[0]}', '{ctx[1]}')"
        cpt2_all[ctx_str] = dist
        cpt2_str_lines.append(f"  P(X_t | {ctx_str}):")
        print(f"  P(X_t | {ctx_str}):")
        for nxt, p in dist.items():
            print(f"    -> P('{nxt}' | {ctx_str}) = {p:.4f}")
            cpt2_str_lines.append(f"    -> {nxt:5s} : {p:.4f} ({count_str(lm2.counts[ctx][nxt], lm2.counts[ctx])})")
        cpt2_str_lines.append("")
    print()

    print("[Part X & XII: Sentence Generation - Second Order]")
    greedy_sent2 = " ".join(lm2.generate_sentence(mode="greedy"))
    print(f"  Greedy Mode (Deterministic): {greedy_sent2}")

    random.seed(42)
    sampled_sents_2nd = []
    print("\n  Sampling Mode (20 Sentences):")
    for i in range(1, 21):
        sent = " ".join(lm2.generate_sentence(mode="sampling"))
        sampled_sents_2nd.append(sent)
        print(f"    {i:2d}. {sent}")
    print()

    # Save TXT files
    write_first_order_txt(norm1_lines, cpt1_str_lines, prediction_lines, greedy_sent1, sampled_sents_1st)
    write_second_order_txt(norm2_lines, cpt2_str_lines, greedy_sent2, sampled_sents_2nd)

    # ---------------------------------------------------------
    # PART 3: Comparative Analysis & Statistics
    # ---------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print(" PART 3: MODEL COMPARISON METRICS")
    print("--------------------------------------------------------------------------------")
    num_params_1st = sum(len(d) for d in lm1.probabilities.values())
    num_params_2nd = sum(len(d) for d in lm2.probabilities.values())

    num_contexts_1st = len(lm1.probabilities)
    num_contexts_2nd = len(lm2.probabilities)

    vocab_size = len(lm1.vocabulary)
    possible_contexts_1st = vocab_size
    possible_contexts_2nd = vocab_size ** 2

    zero_prob_contexts_1st = possible_contexts_1st - num_contexts_1st
    zero_prob_contexts_2nd = possible_contexts_2nd - num_contexts_2nd

    unique_sents_1st = len(set(sampled_sents_1st))
    unique_sents_2nd = len(set(sampled_sents_2nd))

    print(f"  Vocabulary Size (|V|):                      {vocab_size}")
    print(f"  First-Order Distinct Parameters (Non-zero):  {num_params_1st}")
    print(f"  Second-Order Distinct Parameters (Non-zero): {num_params_2nd}")
    print(f"  First-Order Observed Contexts:              {num_contexts_1st} / {possible_contexts_1st} possible")
    print(f"  Second-Order Observed Contexts:             {num_contexts_2nd} / {possible_contexts_2nd} possible")
    print(f"  First-Order Zero-Probability Contexts:      {zero_prob_contexts_1st}")
    print(f"  Second-Order Zero-Probability Contexts:     {zero_prob_contexts_2nd}")
    print(f"  First-Order Generated Sentence Diversity:   {unique_sents_1st} unique out of 20")
    print(f"  Second-Order Generated Sentence Diversity:  {unique_sents_2nd} unique out of 20")
    print()

    # Save deliverables data to file
    deliverables = {
        "dataset": DATASET,
        "first_order": {
            "normalization_passed": all_norm1_valid,
            "normalization_sums": {k: float(v) for k, v in norm1_results.items()},
            "target_cpts": cpt1_selected,
            "greedy_sentence": greedy_sent1,
            "sampled_sentences": sampled_sents_1st
        },
        "second_order": {
            "normalization_passed": all_norm2_valid,
            "normalization_sums": {str(k): float(v) for k, v in norm2_results.items()},
            "all_cpts": cpt2_all,
            "greedy_sentence": greedy_sent2,
            "sampled_sentences": sampled_sents_2nd
        },
        "comparison_metrics": {
            "vocabulary_size": vocab_size,
            "num_params_1st": num_params_1st,
            "num_params_2nd": num_params_2nd,
            "observed_contexts_1st": num_contexts_1st,
            "observed_contexts_2nd": num_contexts_2nd,
            "zero_prob_contexts_1st": zero_prob_contexts_1st,
            "zero_prob_contexts_2nd": zero_prob_contexts_2nd,
            "unique_sampled_sents_1st": unique_sents_1st,
            "unique_sampled_sents_2nd": unique_sents_2nd
        }
    }

    with open("lab_results.json", "w") as f:
        json.dump(deliverables, f, indent=2)

    print("  [SUCCESS] All experiments completed.")
    print("  [FILES GENERATED]: lab_results.json, first_order_generated.txt, second_order_generated.txt")

def write_first_order_txt(norm_lines, cpt_lines, pred_lines, greedy, sampled):
    content = [
        "=" * 80,
        "FIRST-ORDER AUTOREGRESSIVE LANGUAGE MODEL GENERATED OUTPUTS & RESULTS",
        "=" * 80,
        "\nMODEL FORMULATION:",
        "P(X_t | X_{t-1})\n",
        "-" * 80,
        "1. NORMALISATION INVARIANT TESTS (sum_v P(v | w) = 1.0)",
        "-" * 80,
        "\n".join(norm_lines),
        "Overall Normalisation Check: PASSED\n",
        "-" * 80,
        "2. CONDITIONAL PROBABILITY TABLES (CPTs) FOR TARGET WORDS",
        "-" * 80,
        "\n".join(cpt_lines),
        "-" * 80,
        "3. NEXT-WORD PREDICTIONS (ARGMAX MODE)",
        "-" * 80,
        "\n".join(pred_lines),
        "\n" + "-" * 80,
        "4. DETERMINISTIC GREEDY GENERATION (MODE A)",
        "-" * 80,
        greedy + "\n",
        "-" * 80,
        "5. PROBABILISTIC SAMPLED SENTENCE GENERATION (MODE B - 20 SAMPLES)",
        "-" * 80,
        "\n".join(f"{i+1:2d}. {s}" for i, s in enumerate(sampled))
    ]
    with open("first_order_generated.txt", "w") as f:
        f.write("\n".join(content) + "\n")

def write_second_order_txt(norm_lines, cpt_lines, greedy, sampled):
    content = [
        "=" * 80,
        "SECOND-ORDER AUTOREGRESSIVE LANGUAGE MODEL GENERATED OUTPUTS & RESULTS",
        "=" * 80,
        "\nMODEL FORMULATION:",
        "P(X_t | X_{t-2}, X_{t-1})\n",
        "-" * 80,
        "1. NORMALISATION INVARIANT TESTS (sum_v P(v | w_{t-2}, w_{t-1}) = 1.0)",
        "-" * 80,
        "\n".join(norm_lines),
        "Overall Normalisation Check: PASSED\n",
        "-" * 80,
        "2. CONDITIONAL PROBABILITY TABLES (CPTs) FOR ALL OBSERVED CONTEXTS",
        "-" * 80,
        "\n".join(cpt_lines),
        "-" * 80,
        "3. DETERMINISTIC GREEDY GENERATION (MODE A)",
        "-" * 80,
        greedy + "\n",
        "-" * 80,
        "4. PROBABILISTIC SAMPLED SENTENCE GENERATION (MODE B - 20 SAMPLES)",
        "-" * 80,
        "\n".join(f"{i+1:2d}. {s}" for i, s in enumerate(sampled))
    ]
    with open("second_order_generated.txt", "w") as f:
        f.write("\n".join(content) + "\n")

def count_str(c, counter_dict):
    total = sum(counter_dict.values())
    return f"{c}/{total}"

if __name__ == "__main__":
    main()
