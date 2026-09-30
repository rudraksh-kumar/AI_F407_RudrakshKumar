# Bayesian Networks and Autoregressive Language Models Laboratory

A complete Python implementation and theoretical analysis of **First-Order** and **Second-Order Autoregressive Language Models** framed as **Bayesian Networks**, developed for the AI Laboratory.

---

## 📌 Deliverables Mapping (Section 21 of PDF)

The following table maps each required deliverable from the lab specification (`BN_lab.pdf`) to its corresponding file in this repository:

| Required Deliverable (PDF Section 21) | Corresponding File(s) in Repository |
|---|---|
| **1. First-Order Model Python Implementation** | [`first_order_lm.py`](file:///d:/Acads/AI/Lab_BN/first_order_lm.py) |
| **2. Second-Order Model Python Implementation** | [`second_order_lm.py`](file:///d:/Acads/AI/Lab_BN/second_order_lm.py) |
| **3. Conditional Probability Tables (CPTs)** | [`first_order_generated.txt`](file:///d:/Acads/AI/Lab_BN/first_order_generated.txt), [`second_order_generated.txt`](file:///d:/Acads/AI/Lab_BN/second_order_generated.txt), [`lab_results.json`](file:///d:/Acads/AI/Lab_BN/lab_results.json) |
| **4. Examples of Generated Text (Greedy & Sampling)** | [`first_order_generated.txt`](file:///d:/Acads/AI/Lab_BN/first_order_generated.txt), [`second_order_generated.txt`](file:///d:/Acads/AI/Lab_BN/second_order_generated.txt), [`LAB_REPORT.md`](file:///d:/Acads/AI/Lab_BN/LAB_REPORT.md) |
| **5. Probability Normalisation Test Results** | [`first_order_generated.txt`](file:///d:/Acads/AI/Lab_BN/first_order_generated.txt), [`second_order_generated.txt`](file:///d:/Acads/AI/Lab_BN/second_order_generated.txt), [`test_models.py`](file:///d:/Acads/AI/Lab_BN/test_models.py) |
| **6. Answers to Questions 1–14** | [`LAB_REPORT.md`](file:///d:/Acads/AI/Lab_BN/LAB_REPORT.md) |
| **7. Reflection on LLM & Code Inspection Example** | [`LAB_REPORT.md`](file:///d:/Acads/AI/Lab_BN/LAB_REPORT.md#13-part-xv-reflection-on-the-role-of-the-llm--code-validation) |

---

## 📁 Complete List of Files to Submit to GitHub

When committing your repository to GitHub, include **all files** in the project directory:

```text
Lab_BN/
├── BN_lab.pdf                 # Original lab assignment prompt
├── first_order_lm.py          # Deliverable 1: First-order Markov LM implementation
├── second_order_lm.py         # Deliverable 2: Second-order Markov LM implementation
├── first_order_generated.txt  # Deliverables 3-5: First-order CPTs, normalisation tests & 20 generated sentences
├── second_order_generated.txt # Deliverables 3-5: Second-order CPTs, normalisation tests & 20 generated sentences
├── run_experiments.py         # Experiment runner script that trains models and generates outputs
├── test_models.py             # Automated unit test suite (8/8 tests passing)
├── lab_results.json           # Machine-readable JSON dump of all CPTs and test outputs
├── LAB_REPORT.md              # Deliverables 6 & 7: Formal written answers to Questions 1-14 & LLM reflection
└── README.md                  # GitHub repository documentation
```

---

## 🚀 How to Run & Verify

### Prerequisites
- Python 3.8+ (No external ML dependencies required; uses Python standard library).

### 1. Run Automated Unit Tests
```bash
python -m unittest test_models.py
```
*(Runs 8 unit tests checking normalisation invariants \(\sum P = 1.0\), exact CPT values, greedy generation, and sampling validity.)*

### 2. Run Experiments & Regenerate Outputs
```bash
python run_experiments.py
```
*(Trains both models, prints evaluation tables, and updates `first_order_generated.txt`, `second_order_generated.txt`, and `lab_results.json`.)*

---

## 📊 Summary of Model Metrics

| Metric | First-Order Model \(P(X_t \mid X_{t-1})\) | Second-Order Model \(P(X_t \mid X_{t-2}, X_{t-1})\) |
|---|---|---|
| **Context Length (\(k\))** | 1 word | 2 words |
| **Observed Contexts** | 11 / 12 | 14 / 144 |
| **Zero-Probability Contexts** | 1 (8.3%) | 130 (90.3%) |
| **Normalisation Invariant (\(\sum P = 1.0\))** | 100% PASSED | 100% PASSED |
| **Greedy Behavior** | Infinite Loop | Valid Sentence |
| **Grammatical Quality** | Low (allows `"the park"`) | High (100% grammatical) |

---

## 📤 Commands to Submit to GitHub

Run these commands in PowerShell inside `d:\Acads\AI\Lab_BN`:

```powershell
# 1. Initialize Git (if not already initialized)
git init

# 2. Stage all files for submission
git add .

# 3. Commit with a descriptive message
git commit -m "Complete AI Laboratory: Bayesian Networks and Autoregressive Language Models"

# 4. Set main branch and connect to your GitHub repository
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository-name>.git

# 5. Push to GitHub
git push -u origin main
```
