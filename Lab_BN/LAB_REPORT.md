# AI Laboratory Report: Bayesian Networks & Autoregressive Language Models

**Author:** Student  
**Course:** AI Laboratory  
**Topic:** Bayesian Networks, Chain Rule Factorisation, First-Order and Second-Order Autoregressive Language Models  
**Repository:** [Lab_BN](file:///d:/Acads/AI/Lab_BN)  

---

## Executive Summary

This laboratory explores the fundamental probabilistic foundation of modern Autoregressive Language Models (LLMs) through the lens of Bayesian Networks. By factorising sentence probability distributions using the chain rule of probability, we construct first-order and second-order Markov language models from raw text data using explicit Conditional Probability Tables (CPTs). We validate probability normalisation invariants, implement greedy and probabilistic sampling text generation algorithms, analyze context length trade-offs, reflect on LLM-assisted code generation with explicit code inspection/correction examples, and compare simple Bayesian network models with modern neural language architectures.

---

## 1. Overview & Theoretical Framework

Autoregressive language models generate text by predicting the next word or token given the preceding sequence of words. This process relies on the **chain rule of probability**, which factorises the joint probability of a sequence of tokens \(X_1, X_2, \dots, X_T\):

\[
P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^{T} P(X_t \mid X_1, \dots, X_{t-1})
\]

Where:
- \(X_t\) represents the random variable for the token at position \(t\).
- \(P(X_t \mid X_1, \dots, X_{t-1})\) represents the conditional probability of the next token given all previous tokens.

---

## 2. Part I: From Probability to Language

Consider the sentence: `"the cat sat on the mat"`.
Assigning random variables:
\[
X_1 = \text{the}, \quad X_2 = \text{cat}, \quad X_3 = \text{sat}, \quad X_4 = \text{on}, \quad X_5 = \text{the}, \quad X_6 = \text{mat}
\]

Using the chain rule without any independence assumptions:
\[
P(X_1, \dots, X_6) = P(X_1) P(X_2 \mid X_1) P(X_3 \mid X_1, X_2) P(X_4 \mid X_1, X_2, X_3) P(X_5 \mid X_1, \dots, X_4) P(X_6 \mid X_1, \dots, X_5)
\]

### Question 1 Response
> **Question 1:** *Why is this decomposition useful for generating text?*

**Answer:**  
1. **Intractability Reduction:** Directly estimating a high-dimensional joint probability distribution \(P(X_1, \dots, X_T)\) over all possible complete sentences of length \(T\) requires counting every conceivable sentence in existence, which is computationally intractable and suffers from severe data sparsity.
2. **Sequential Autoregressive Generation:** The chain-rule factorisation converts the joint estimation problem into a sequence of step-by-step local conditional decisions \(P(X_t \mid \text{context})\). Text generation becomes an iterative process: at step \(t\), the model samples a token \(X_t\) from the conditional distribution, appends \(X_t\) to the sequence, and uses the updated sequence as the context for step \(t+1\).

---

## 3. Part II: A First-Order Bayesian Network for Text

A **first-order Markov language model** introduces a conditional independence assumption: each token \(X_t\) depends **only** on the immediately preceding token \(X_{t-1}\).

Schematically represented as a Directed Acyclic Graph (DAG):
\[
X_1 \longrightarrow X_2 \longrightarrow X_3 \longrightarrow \dots \longrightarrow X_T
\]

The joint probability simplifies to:
\[
P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^{T} P(X_t \mid X_{t-1})
\]

### Question 2 Response
> **Question 2:** *What independence assumption is being made by this network? Express your answer using probability notation.*

**Answer:**  
The first-order Markov assumption states that token \(X_t\) is conditionally independent of all historical tokens preceding \(X_{t-1}\) (i.e., \(X_1, X_2, \dots, X_{t-2}\)), given the immediate predecessor \(X_{t-1}\).

**Probability Notation:**
\[
P(X_t \mid X_1, X_2, \dots, X_{t-1}) = P(X_t \mid X_{t-1})
\]

**Conditional Independence Notation:**
\[
X_t \perp\!\!\!\perp (X_1, X_2, \dots, X_{t-2}) \mid X_{t-1}
\]

---

## 4. Part III: Dataset & Preprocessing

The laboratory utilizes a small structured corpus of 6 sentences:

```text
1. the cat sat on the mat
2. the cat sat on the rug
3. the dog sat on the mat
4. the dog ran to the park
5. the cat ran to the park
6. the dog sat on the rug
```

### Tokenization Strategy
- Text is converted to lowercase.
- Special boundary tokens `<START>` and `<END>` are prepended and appended to each sentence.

**Tokenized Sentences:**
1. `<START> the cat sat on the mat <END>`
2. `<START> the cat sat on the rug <END>`
3. `<START> the dog sat on the mat <END>`
4. `<START> the dog ran to the park <END>`
5. `<START> the cat ran to the park <END>`
6. `<START> the dog sat on the rug <END>`

**Vocabulary (\(V\)):** `{'<START>', 'the', 'cat', 'sat', 'on', 'mat', 'rug', 'dog', 'ran', 'to', 'park', '<END>'}` (\(|V| = 12\)).

---

## 5. Part IV & V: Conditional Probability Table (CPT) Construction

The conditional probability \(P(w_j \mid w_i)\) is computed via Maximum Likelihood Estimation (MLE) by counting transition frequencies:

\[
P(w_j \mid w_i) = \frac{C(w_i, w_j)}{\sum_{k} C(w_i, w_k)}
\]

### Question 3 Response & Full CPT Analysis
> **Question 3:** *Construct the conditional probability distribution \(P(\text{next word} \mid \text{current word})\) for at least `the`, `cat`, `dog`, `sat`, `ran`. Identify any zero-probability transitions.*

#### Transition Counts & Computed CPTs:

| Current Word (\(w_i\)) | Outgoing Transitions & Counts | Conditional Probabilities \(P(X_t \mid w_i)\) | Zero-Probability Next Tokens |
|---|---|---|---|
| **`the`** (Count = 12) | `cat`: 3, `dog`: 3, `mat`: 2, `rug`: 2, `park`: 2 | \(P(\text{cat}) = 0.2500\)<br>\(P(\text{dog}) = 0.2500\)<br>\(P(\text{mat}) = 0.1667\)<br>\(P(\text{rug}) = 0.1667\)<br>\(P(\text{park}) = 0.1667\) | `<START>`, `the`, `sat`, `on`, `ran`, `to` (6 / 11 zero) |
| **`cat`** (Count = 3) | `sat`: 2, `ran`: 1 | \(P(\text{sat}) = 0.6667\)<br>\(P(\text{ran}) = 0.3333\) | 9 / 11 zero |
| **`dog`** (Count = 3) | `sat`: 2, `ran`: 1 | \(P(\text{sat}) = 0.6667\)<br>\(P(\text{ran}) = 0.3333\) | 9 / 11 zero |
| **`sat`** (Count = 4) | `on`: 4 | \(P(\text{on}) = 1.0000\) | 10 / 11 zero |
| **`ran`** (Count = 2) | `to`: 2 | \(P(\text{to}) = 1.0000\) | 10 / 11 zero |
| **`<START>`** (Count = 6) | `the`: 6 | \(P(\text{the}) = 1.0000\) | 10 / 11 zero |
| **`on`** (Count = 4) | `the`: 4 | \(P(\text{the}) = 1.0000\) | 10 / 11 zero |
| **`to`** (Count = 2) | `the`: 2 | \(P(\text{the}) = 1.0000\) | 10 / 11 zero |
| **`mat`** (Count = 2) | `<END>`: 2 | \(P(\text{<END>}) = 1.0000\) | 10 / 11 zero |
| **`rug`** (Count = 2) | `<END>`: 2 | \(P(\text{<END>}) = 1.0000\) | 10 / 11 zero |
| **`park`** (Count = 2) | `<END>`: 2 | \(P(\text{<END>}) = 1.0000\) | 10 / 11 zero |

---

## 6. Part VI: Code Inspection & Specification

Refer to [`first_order_lm.py`](file:///d:/Acads/AI/Lab_BN/first_order_lm.py) for the full source code.

### Question 4 Response
> **Question 4:** *Where in the program are the transition counts stored?*

**Answer:**  
Transition counts are stored in the instance variable `self.counts`, initialized as `defaultdict(Counter)`.  
- Accessing `self.counts[w_i][w_j]` returns integer count \(C(w_i, w_j)\).

### Question 5 Response
> **Question 5:** *Where is \(P(X_t \mid X_{t-1})\) computed?*

**Answer:**  
In the `train()` method of [`FirstOrderLanguageModel`](file:///d:/Acads/AI/Lab_BN/first_order_lm.py#L18-L38):
```python
for w1, next_counts in self.counts.items():
  total_transitions = sum(next_counts.values())
  for w2, count in next_counts.items():
    self.probabilities[w1][w2] = count / total_transitions
```

### Question 6 Response
> **Question 6:** *How does the program choose the next word? Is it: 1. always choosing the most probable word, or 2. sampling from the probability distribution? Explain the difference.*

**Answer:**  
The implementation supports both selection mechanisms:
1. **Greedy Selection (`predict_next_greedy`)**: Computes \(\arg\max_w P(w \mid w_{\text{prev}})\). Deterministic.
2. **Sampling (`sample_next`)**: Uses `random.choices(tokens, weights=probs)` to sample stochastically based on the multinomial probability distribution \(P(w \mid w_{\text{prev}})\).
- **Key Difference**: Greedy generation always picks the single mode of the distribution, leading to deterministic outputs. Sampling reflects the underlying probability distribution over repeated trials, allowing diverse sentence paths.

### Question 7 Response
> **Question 7:** *What happens if the program encounters a word for which no transition has been observed?*

**Answer:**  
In our implementation, if `get_distribution(prev_token)` encounters an unobserved token, it returns an empty dictionary `{}`. The generation logic intercepts this empty distribution and returns the `<END>` token, terminating generation safely. Without this check, calling `max()` or `random.choices()` on an empty collection would throw a `ValueError`.

---

## 7. Part VII: Model Testing & Normalisation Invariants

A fundamental invariant of any probability distribution is that the sum of probabilities over all mutually exclusive outcomes in the sample space must equal exactly 1.0:

\[
\sum_{v \in V} P(v \mid w) = 1.0 \quad \forall w \in V
\]

### Normalisation Test Results ([`run_experiments.py`](file:///d:/Acads/AI/Lab_BN/run_experiments.py#L46-L55))

```text
[Normalisation Test - First Order Model]
  P(* | '<START>' ) sum = 1.000000 [VALID]
  P(* | 'the'     ) sum = 1.000000 [VALID]
  P(* | 'cat'     ) sum = 1.000000 [VALID]
  P(* | 'sat'     ) sum = 1.000000 [VALID]
  P(* | 'on'      ) sum = 1.000000 [VALID]
  P(* | 'mat'     ) sum = 1.000000 [VALID]
  P(* | 'rug'     ) sum = 1.000000 [VALID]
  P(* | 'dog'     ) sum = 1.000000 [VALID]
  P(* | 'ran'     ) sum = 1.000000 [VALID]
  P(* | 'to'      ) sum = 1.000000 [VALID]
  P(* | 'park'    ) sum = 1.000000 [VALID]
  Overall Normalisation Check: PASSED
```

### Question 8 Response
> **Question 8:** *If one of the totals is 0.87, what does this tell you about the implementation?*

**Answer:**  
A total of 0.87 indicates a **bug in probability normalization logic**. Specifically:
1. One or more valid transition outcomes were omitted from the numerator sum while maintaining a larger denominator.
2. The denominator was computed over a global vocabulary constant rather than the exact sum of observed outgoing transition counts.
3. Floating-point truncation or improper frequency updates occurred.  
In a valid probabilistic system, \(\sum_v P(v \mid w)\) must sum to 1.0 within numerical float precision.

---

## 8. Part VIII & IX: Prediction and Sentence Generation

### Question 9 Response
> **Question 9:** *Are the most probable predictions always the same as the words that you would personally expect? What does this tell you about the difference between a probability model and human linguistic expectations?*

**Answer:**  
No. In the first-order model, after `the`, the model assigns equal probability (0.25) to `cat` and `dog`, and 0.1667 to `mat`, `rug`, and `park`. More critically, after `on`, the model predicts `the` (1.0), and after `the`, it can predict `cat` or `dog`, yielding nonsensical strings like `"sat on the cat"`.
- **Insight:** A simple n-gram probability model relies strictly on raw co-occurrence frequencies in training data. It lacks semantic comprehension, world knowledge, and structural grammar. Human expectations rely on deep compositional semantics and syntactic structure.

### Sentence Generation Results (First-Order Model)

- **Greedy Generation (Deterministic Mode A):**
  `<START> the cat sat on the cat sat on the cat sat on the cat...`  
  *(Stuck in an infinite 4-gram cyclic loop!)*

- **Sampling Generation (Probabilistic Mode B - 20 Samples):**
  ```text
  1. <START> the cat sat on the dog ran to the cat sat on the cat sat on the dog ran to the dog sat on the mat <END>
  2. <START> the park <END>
  3. <START> the dog sat on the rug <END>
  4. <START> the park <END>
  5. <START> the cat sat on the cat sat on the mat <END>
  6. <START> the mat <END>
  7. <START> the dog sat on the mat <END>
  8. <START> the rug <END>
  9. <START> the dog sat on the mat <END>
  10. <START> the park <END>
  ...
  ```

> [!NOTE]
> In First-Order Sampling, sentences like `<START> the park <END>` or `<START> the mat <END>` are generated! This occurs because `the` is followed by `park` (prob 0.1667), and `park` is followed by `<END>` (prob 1.0). The first-order model forgets that `park` only occurred after `to the` in the original dataset!

---

## 9. Part X: Deterministic vs Probabilistic Generation

### Question 10 Response
> **Question 10:** *Compare the two sets of generated sentences. Which mode produces more variation? Why?*

**Answer:**  
- **Sampling mode** produces far more variation (10 unique sentences generated across 20 trials).
- **Greedy mode** produces 0 variation (1 unique deterministic output).
- **Why?** Greedy generation deterministically picks \(\arg\max_w P(w \mid \text{context})\) at every step. Because the mode of the conditional distribution is fixed for each context, the model follows the identical transition path every time. Sampling draws stochastically from the probability vector, exploring all non-zero paths in the transition graph.

---

## 10. Part XI & XII: Second-Order Bayesian Network

To solve the context loss of first-order models, we construct a **Second-Order Markov Language Model**:

\[
P(X_t \mid X_1, \dots, X_{t-1}) \approx P(X_t \mid X_{t-2}, X_{t-1})
\]

Schematic Bayesian Network DAG structure:
\[
X_{t-2} \longrightarrow X_t \longleftarrow X_{t-1}
\]

For 4 tokens \(X_1, X_2, X_3, X_4\):
\[
P(X_1, X_2, X_3, X_4) = P(X_1) P(X_2 \mid X_1) P(X_3 \mid X_1, X_2) P(X_4 \mid X_2, X_3)
\]

### Question 11 Response
> **Question 11:** *How does the second-order model differ from the first-order model in terms of: 1. graph structure, 2. CPT, 3. context available, 4. data needed?*

**Answer:**  
1. **Graph Structure:** First-order is a linear chain \(X_{t-1} \to X_t\). Second-order has dual parents \((X_{t-2}, X_{t-1}) \to X_t\) for every node \(X_t\) (\(t \ge 3\)).
2. **Conditional Probability Table (CPT):** First-order CPT is a 2D matrix of shape \(|V| \times |V|\). Second-order CPT is a 3D tensor of shape \(|V| \times |V| \times |V|\) (mapping context tuples \((w_{t-2}, w_{t-1})\) to next token probabilities).
3. **Context Available:** First-order uses 1 preceding word of memory. Second-order uses 2 preceding words of memory.
4. **Data Needed:** Second-order requires significantly more training data (exponential growth with context length) to reliably estimate counts for all \(|V|^2\) context pairs.

---

## 11. Part XIII: Model Comparison & Results

Refer to [`second_order_lm.py`](file:///d:/Acads/AI/Lab_BN/second_order_lm.py) for the complete implementation.

### Second-Order Generated Output

- **Greedy Mode (Deterministic):**
  `<START> the cat sat on the mat <END>`  
  *(Successfully generates a completely grammatical, valid sentence without looping!)*

- **Sampling Mode (20 Samples):**
  ```text
  1. <START> the dog sat on the rug <END>
  2. <START> the dog sat on the mat <END>
  3. <START> the cat sat on the mat <END>
  4. <START> the dog sat on the mat <END>
  5. <START> the dog sat on the rug <END>
  6. <START> the dog ran to the park <END>
  ...
  ```
  *(100% of generated sentences are valid, grammatical sentences from the true language distribution!)*

### Question 12 Response
> **Question 12:** *Why does increasing the amount of context potentially improve prediction? Why can it simultaneously make the model harder to estimate from limited data? Relate your answer to the size of the conditional probability table.*

**Answer:**  
- **Improvement:** Increasing context length disambiguates history. In our dataset, knowing `on the` precedes `mat` or `rug`, while `to the` precedes `park`, prevents ungrammatical combinations like `"to the mat"` or `"on the park"`.
- **Estimation Difficulty (Curse of Dimensionality):** The size of the CPT grows exponentially as \(O(|V|^{k+1})\), where \(k\) is the context length. With vocabulary \(|V| = 12\), first-order has \(|V| = 12\) contexts, whereas second-order has \(|V|^2 = 144\) context tuples. In our dataset, only 14 context pairs were observed, leaving **130 zero-probability contexts** (90.3% sparsity).

### Quantitative Metrics Comparison Table

| Metric | First-Order Model | Second-Order Model |
|---|---|---|
| **Context Length (\(k\))** | 1 word (\(X_{t-1}\)) | 2 words (\(X_{t-2}, X_{t-1}\)) |
| **Total Possible Contexts (\(|V|^k\))** | 12 | 144 |
| **Observed Contexts** | 11 | 14 |
| **Zero-Probability Contexts** | 1 (8.3%) | 130 (90.3%) |
| **Distinct Parameters (Non-zero)** | 17 | 18 |
| **Normalisation Tests Passed** | 100% (11/11) | 100% (14/14) |
| **Greedy Behavior** | Infinite Loop | Valid Complete Sentence |
| **Grammaticality of Sampled Output** | Low (produces `"the park"`) | 100% Valid Sentences |
| **Generated Diversity (20 samples)** | 10 unique | 6 unique |

---

## 12. Part XIV: Connection to Modern Language Models

Modern LLMs (e.g. GPT-4, Gemini) are also autoregressive models that factorise sequence probabilities using the chain rule:

\[
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})
\]

### Comparison: Simple BN Model vs. Modern Autoregressive LLM

| Feature | Simple Bayesian Network Model | Modern Autoregressive LLM |
|---|---|---|
| **Distribution Representation** | Explicit Conditional Probability Tables (CPTs) | Deep Neural Network (Transformer) |
| **Context Handling** | Fixed N-gram window (\(k=1\) or \(k=2\)) | Large context window (8k - 1M+ tokens) via Self-Attention |
| **Parameter Storage** | Explicit transition counts/frequencies | Billions of continuous learned neural weights |
| **Learning Algorithm** | Direct frequency counting (MLE) | Gradient-based backpropagation (AdamW, Cross-Entropy Loss) |
| **Generalization** | Fails on unseen contexts (zero-probability) | Dense embeddings generalize to unseen contexts |
| **Generation Technique** | Sampling / Greedy | Temperature sampling, Top-k, Top-p (Nucleus) sampling, Beam Search |

---

## 13. Part XV: Reflection on the Role of the LLM & Code Validation

### Question 13 Response
> **Question 13:** *Why is Approach B ("Implement probabilistic model P(X_t | X_{t-1}) from transition counts with sampling") preferable to Approach A ("Write a Python language model for me")?*

**Answer:**  
1. **Specifying Intended Behaviour:** Prompting with explicit mathematical specifications ensures the generated code implements the exact targeted probabilistic model rather than returning generic or black-box neural libraries.
2. **Understanding Representation:** Forces developers to inspect how CPTs, state transitions, and counts are structured in memory.
3. **Validating Generated Implementation:** Provides clear criteria against which code can be verified.
4. **Testing Probabilistic Invariants:** Enables writing formal unit tests (e.g., probability normalisation \(\sum P = 1.0\)).
5. **Distinguishing Implementation from Model:** Clarifies that Python dictionaries are software constructs, whereas the Bayesian network is the underlying probabilistic model.

### Code Inspection & Correction Example (Deliverable #7)

During the inspection of LLM-generated code for the second-order model, an issue was discovered in handling unobserved context tuples during text sampling.

```diff
- # Uncorrected LLM Code Snippet:
- def sample_next(self, context_tuple):
-     dist = self.probabilities[context_tuple]  # Raises KeyError for unobserved contexts!
-     tokens = list(dist.keys())
-     probs = list(dist.values())
-     return random.choices(tokens, weights=probs)[0]

+ # Corrected Implementation (in second_order_lm.py):
+ def sample_next(self, context_tuple):
+     dist = self.get_distribution(context_tuple)
+     if not dist:
+         return self.end_token  # Gracefully handles zero-probability/unobserved contexts!
+     tokens, probs = zip(*dist.items())
+     return random.choices(tokens, weights=probs, k=1)[0]
```

**Rationale:** The initial uncorrected snippet assumed every context pair would exist in `self.probabilities`. When sampling encountering an unseen context pair, it would crash with a `KeyError`. The corrected version gracefully returns `<END>`, satisfying the probabilistic stopping condition.

---

## 14. Part XVI: Final Question - What Did the Bayesian Network Add?

### Question 14 Response
> **Question 14:** *What did thinking of the language model as a Bayesian network give you? Discuss at least three key aspects.*

**Answer:**  
1. **Factorisation of Joint Distribution:** Provided a clear mathematical foundation for decomposing sequence probabilities into local conditional probabilities using DAG properties.
2. **Reasoning About Independence Assumptions:** Clearly highlighted how Markov assumptions simplify computation and showed the precise structural tradeoff between context length and graph complexity.
3. **Testing Probabilistic Invariants:** Allowed us to establish rigorous testable properties (such as \(\sum_v P(v \mid \text{context}) = 1.0\)) to verify implementation correctness.

---

## 15. Deliverables & Artifacts Summary

All required deliverables have been implemented and verified in the repository:

1. [`first_order_lm.py`](file:///d:/Acads/AI/Lab_BN/first_order_lm.py): First-order autoregressive model class.
2. [`second_order_lm.py`](file:///d:/Acads/AI/Lab_BN/second_order_lm.py): Second-order autoregressive model class.
3. [`run_experiments.py`](file:///d:/Acads/AI/Lab_BN/run_experiments.py): Comprehensive test and execution script.
4. [`test_models.py`](file:///d:/Acads/AI/Lab_BN/test_models.py): Automated unit test suite (8/8 tests passing).
5. [`lab_results.json`](file:///d:/Acads/AI/Lab_BN/lab_results.json): Structured JSON dump of all CPT tables, normalisation test outputs, generated sentences, and model metrics.
6. [`LAB_REPORT.md`](file:///d:/Acads/AI/Lab_BN/LAB_REPORT.md): This complete laboratory report.
