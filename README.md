# Artificial Intelligence (AI_F407) — Laboratory Solutions & Reports

Welcome to the comprehensive repository for **AI_F407 Artificial Intelligence**. This repository contains complete code implementations, experimental benchmarks, and structured laboratory reports across search algorithms, agent design, logic & automated planning, neural network architectures, and probabilistic graphical models.

---

## 📌 Table of Contents

1. [Warehouse Robot Navigation (Search & A*)](#1-warehouse-robot-navigation-search--a)
2. [Autonomous Agent Architecture](#2-autonomous-agent-architecture)
3. [Knowledge Representation & Logic Planning](#3-knowledge-representation--logic-planning)
4. [Neural Language Models](#4-neural-language-models)
5. [Bayesian Networks & Probabilistic Reasoning](#5-bayesian-networks--probabilistic-reasoning)
6. [Repository Structure](#-repository-structure)

---

## 1. Warehouse Robot Navigation (Search & A*)

* **Folder Directory:** [`Search/`](./Search/)
* **Problem Specification:** [`Search/search_lab_ex.pdf`](./Search/search_lab_ex.pdf)
* **Python Implementation:** [`Search/warehouse_search.py`](./Search/warehouse_search.py)
* **Laboratory Report:** [`Search/submission.md`](./Search/submission.md)

### 💡 Core Concepts & Implementation
Formulates warehouse robot navigation as a formal search problem $\mathcal{P} = (S, A, T, s_0, G, c)$ on a 2D ASCII grid map.
* **Algorithms Implemented:** A* Search (Min-Heap via `heapq`) and Uninformed Breadth-First Search (BFS).
* **Heuristic Analysis:** Benchmarks Manhattan Distance ($|dx| + |dy|$), Zero Heuristic ($h(n)=0$), Euclidean Distance, and Weighted Inadmissible Manhattan Distance ($2 \times h$).
* **Key Findings:** Empirical comparison demonstrating state expansion metrics, path length optimality (40 steps), and how maze corridor geometry impacts frontier pruning.

---

## 2. Autonomous Agent Architecture

* **Folder Directory:** [`Agents/`](./Agents/)
* **Problem Specification:** [`Agents/agents_lab.pdf`](./Agents/agents_lab.pdf)
* **Python Implementation:** [`Agents/warehouse_agent.py`](./Agents/warehouse_agent.py)
* **Laboratory Report:** [`Agents/SUBMISSION.md`](./Agents/SUBMISSION.md)

### 💡 Core Concepts & Implementation
Explores goal-based and utility-based agent design operating in dynamic grid environments.
* **Key Components:** State representation, action selection mechanisms, goal recognition, and memory state management.
* **Agent Evaluation:** Evaluates path cost efficiency, state exploration overhead, and safe navigation behavior around static grid obstacles.

---

## 3. Knowledge Representation & Logic Planning

* **Folder Directory:** [`Logic/`](./Logic/)
* **Problem Specification:** [`Logic/logic_lab_ex (1).pdf`](./Logic/logic_lab_ex%20(1).pdf)
* **Code Solutions:**
  * Prolog Planner: [`Logic/planner.pl`](./Logic/planner.pl)
  * Python Verifier: [`Logic/verify_prolog.py`](./Logic/verify_prolog.py)
  * Python Planner: [`Logic/planner.py`](./Logic/planner.py)
* **Laboratory Reports:** [`Logic/submission.md`](./Logic/submission.md) \| [`Logic/lab_report.md`](./Logic/lab_report.md)

### 💡 Core Concepts & Implementation
Covers First-Order Logic (FOL), resolution theorem proving, and automated STRIPS-style planning.
* **Prolog & Python Integration:** Implements logic rules and goal verification in Prolog (`planner.pl`) and validates model output programmatically using Python verification pipelines.

---

## 4. Neural Language Models

* **Folder Directory:** [`Neural_Models/`](./Neural_Models/)
* **Python Pipeline:** [`Neural_Models/neural_models_lab.py`](./Neural_Models/neural_models_lab.py)
* **Laboratory Report:** [`Neural_Models/report_submission.md`](./Neural_Models/report_submission.md)

### 💡 Core Concepts & Implementation
Implements neural network architectures for sequence modeling and classification tasks.
* **Pipeline Design:** Data preprocessing, embedding layers, forward pass loss calculation, backpropagation optimization, and evaluation metrics logging.

---

## 5. Bayesian Networks & Probabilistic Reasoning

* **Folder Directory:** [`Bayesian_Networks/`](./Bayesian_Networks/)
* **Problem Specification:** [`Bayesian_Networks/BN_lab.pdf`](./Bayesian_Networks/BN_lab.pdf)
* **Code Solutions:**
  * Experiment Pipeline: [`Bayesian_Networks/run_experiments.py`](./Bayesian_Networks/run_experiments.py)
  * First-Order LM: [`Bayesian_Networks/first_order_lm.py`](./Bayesian_Networks/first_order_lm.py)
  * Second-Order LM: [`Bayesian_Networks/second_order_lm.py`](./Bayesian_Networks/second_order_lm.py)
* **Laboratory Report:** [`Bayesian_Networks/LAB_REPORT.md`](./Bayesian_Networks/LAB_REPORT.md)
* **Experimental Output:** [`Bayesian_Networks/lab_results.json`](./Bayesian_Networks/lab_results.json)

### 💡 Core Concepts & Implementation
Focuses on probabilistic graphical models, conditional independence, and Markovian text generation models.
* **Language Modeling Experiments:** Trains 1st-order and 2nd-order Markov language models, analyzing text generation perplexity, state transition matrices, and empirical probability distribution convergence.

---

## 📂 Repository Structure

```text
AI_F407_RudrakshKumar/
├── Search/
│   ├── search_lab_ex.pdf       # Problem Specification
│   ├── warehouse_search.py     # A* & BFS Search Implementation
│   └── submission.md           # Lab Report
├── Agents/
│   ├── agents_lab.pdf          # Problem Specification
│   ├── warehouse_agent.py      # Autonomous Agent Code
│   └── SUBMISSION.md           # Lab Report
├── Logic/
│   ├── logic_lab_ex (1).pdf    # Problem Specification
│   ├── planner.pl              # Prolog Planner
│   ├── planner.py              # Python Planner
│   ├── verify_prolog.py        # Logic Verifier
│   └── submission.md           # Lab Report
├── Neural_Models/
│   ├── neural_models_lab.py    # Deep Learning Pipeline
│   └── report_submission.md    # Lab Report
└── Bayesian_Networks/
    ├── BN_lab.pdf              # Problem Specification
    ├── first_order_lm.py       # 1st-Order Markov Model
    ├── second_order_lm.py      # 2nd-Order Markov Model
    ├── run_experiments.py      # Benchmark Script
    └── LAB_REPORT.md           # Lab Report
