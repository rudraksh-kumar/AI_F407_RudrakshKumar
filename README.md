# Laboratory: Logical Reasoning for Planning

This repository contains the complete implementation, plan verification, and laboratory documentation for **Logical Reasoning for Planning**, exploring how logical deduction and search work together to create and verify planning agents.

---

## 📌 Overview & Core Concepts

A planning problem asks an agent to find a sequence of actions that transforms an initial state $I$ into a desired goal state $G$, represented as $(I, A, G)$.

$$\text{Logic} + \text{Search} = \text{Planning}$$

- **Logical Reasoning:** Evaluates action applicability ($S \models \text{Preconditions}(a)$) and computes deterministic state transitions ($S' = \text{Apply}(S, a) = (S \setminus \text{Effects}^-(a)) \cup \text{Effects}^+(a)$).
- **Search (BFS):** Explores candidate sequences level-by-level to locate optimal, shortest-path plans reaching goal state $G$.
- **Independent Verification:** Uses Prolog (`planner.pl`) as a formal declarative verifier to independently check AI-generated plans (*Generate $\rightarrow$ Independent Verification* paradigm).

---

## 📁 Repository Structure

```text
.
├── planner.py            # Python implementation of BFS Planning Agent & test suite (Tests A, B, C)
├── planner.pl            # SWI-Prolog knowledge base, movement rules, plan verifier & Task 8 logic
├── verify_prolog.py      # Independent Python script simulating Prolog query resolution & verification
├── lab_report.md         # Comprehensive markdown report addressing all tasks and PDF reflection questions
├── submission.md         # Formatted complete laboratory submission document
├── logic_lab_ex (1).pdf  # Original laboratory specification document
└── README.md             # Project documentation & GitHub submission guide
```

---

## 🚀 Getting Started & Execution

### Prerequisites
- **Python 3.8+**
- **SWI-Prolog** (Optional, for running Prolog queries directly via `swipl`)

### 1. Run Python Planner Agent & Test Suite
Executes Test A (solvable warehouse problem), Test B (impossible problem with PickUp removed), and Test C (irrelevant actions added).

```bash
python planner.py
```

#### Output Example:
```text
--- Test A: Solvable Problem ---
Plan found!
Initial State: {'At(Package, A)', 'At(Robot, A)'}
Step 1: PickUp(Package, A)
  State reached: {'At(Robot, A)', 'Holding(Package)'}
Step 2: Move(A, B)
  State reached: {'At(Robot, B)', 'Holding(Package)'}
Step 3: Move(B, C)
  State reached: {'Holding(Package)', 'At(Robot, C)'}
Step 4: Drop(Package, C)
  State reached: {'At(Package, C)', 'At(Robot, C)'}
```

### 2. Run Prolog Verification Simulation
Simulates SWI-Prolog resolution for Task 6 (`can_move`), Task 7 (`valid_plan`), and Task 8 (`reduce_speed` deduction chain).

```bash
python verify_prolog.py
```

### 3. Run SWI-Prolog (Native Prolog CLI)
If SWI-Prolog is installed, load `planner.pl` and run queries:

```bash
swipl planner.pl
```

```prolog
?- can_move(a, b).
% Returns: true

?- valid_plan([move(a,b), move(b,c)]).
% Returns: true

?- valid_move(a, c).
% Returns: false (Shortcut rejected!)

?- reduce_speed.
% Returns: true
```

---

## 📑 Summary of Tasks & Results

| Task | Description | Status / Result |
| :--- | :--- | :--- |
| **Task 0** | Problem Specification & Precondition Analysis | Completed ($I, A, G$ formally specified; `PickUp` applicable, `Drop` inapplicable) |
| **Task 1** | Hand-Constructed Plan | Completed (4-step trace $S_0 \to S_1 \to S_2 \to S_3 \to S_4$) |
| **Task 2 & 3** | Python BFS Planner Implementation & Testing | Verified (`planner.py` tested against Test A, Test B, and Test C) |
| **Task 4** | Logic & Search Architecture Mapping | Completed (Completed sequence diagram & co-operation rationale) |
| **Task 5** | LLM Self-Verification Analysis | Completed (*Generated explanation $\neq$ Independent verification*) |
| **Section 5**| Reflection Questions (PDF Page 10) | All 7 questions answered in detail |
| **Task 6** | Prolog as a Plan Verifier | Completed (`can_move/2` rules & CWA resolution explained) |
| **Task 7** | Prolog Proposed Plan Check & Challenge | Completed (`valid_plan/1` verifies valid moves, rejects `Move(a,c)`) |
| **Task 8** | Connect Prolog to Logical Reasoning | Completed ($\text{wet\_road} \implies \text{slippery} \implies \text{reduce\_speed}$ implication chain) |
| **Section 7.2**| Prolog Reflection Questions (PDF Page 14)| All 4 questions answered |

---

## 🔑 Key Engineering Takeaways

1. **Logic determines what is possible; search determines what to try.**
   Without preconditions, search space grows exponentially with illegal states. Logic bounds the search to valid transitions.
2. **Generate $\rightarrow$ Independent Verification:**
   An AI system (such as an LLM or Python search agent) can rapidly propose candidate solution plans, while a separate formal logical system (such as Prolog) independently validates them against strict domain constraints.
