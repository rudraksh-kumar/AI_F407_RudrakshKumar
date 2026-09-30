# Laboratory Submission: Logical Reasoning for Planning

## 1. Specification of the Planning Problem (Task 0)

### (a) Initial State ($I$)
$$I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$$

### (b) Goal State ($G$)
$$G = \{\text{At}(\text{Package}, C)\}$$

### (c) Available Actions ($A$)
- $\text{Move}(X, Y)$
- $\text{PickUp}(\text{Package}, X)$
- $\text{Drop}(\text{Package}, X)$

### (d) Action Preconditions and Effects

1. **$\text{Move}(X, Y)$**
   - **Positive Preconditions:** $\{\text{At}(\text{Robot}, X)\}$
   - **Negative Preconditions:** $\emptyset$
   - **Positive Effects:** $\{\text{At}(\text{Robot}, Y)\}$
   - **Negative Effects:** $\{\text{At}(\text{Robot}, X)\}$

2. **$\text{PickUp}(\text{Package}, X)$**
   - **Positive Preconditions:** $\{\text{At}(\text{Robot}, X), \text{At}(\text{Package}, X)\}$
   - **Negative Preconditions:** $\emptyset$
   - **Positive Effects:** $\{\text{Holding}(\text{Package})\}$
   - **Negative Effects:** $\{\text{At}(\text{Package}, X)\}$

3. **$\text{Drop}(\text{Package}, X)$**
   - **Positive Preconditions:** $\{\text{At}(\text{Robot}, X), \text{Holding}(\text{Package})\}$
   - **Negative Preconditions:** $\emptyset$
   - **Positive Effects:** $\{\text{At}(\text{Package}, X)\}$
   - **Negative Effects:** $\{\text{Holding}(\text{Package})\}$

### Task 0 Question: Action Applicability Analysis
- **Is `PickUp(Package, A)` applicable initially?**
  - **Yes.** Its preconditions are $\{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$. Both propositions are present in the initial state $I$.
- **What about `Drop(Package, C)`?**
  - **No.** Its preconditions are $\{\text{At}(\text{Robot}, C), \text{Holding}(\text{Package})\}$. Neither proposition is true in the initial state $I$.

---

## 2. Manually Constructed Plan (Task 1)

Below is the state-by-state execution trace for transporting the package from $A$ to $C$:

| State | Facts in Knowledge Base | Action Executed |
| :--- | :--- | :--- |
| **$S_0$** | `At(Robot, A)`, `At(Package, A)` | `PickUp(Package, A)` |
| **$S_1$** | `At(Robot, A)`, `Holding(Package)` | `Move(A, B)` |
| **$S_2$** | `At(Robot, B)`, `Holding(Package)` | `Move(B, C)` |
| **$S_3$** | `At(Robot, C)`, `Holding(Package)` | `Drop(Package, C)` |
| **$S_4$** | `At(Robot, C)`, `At(Package, C)` | *Goal Achieved ($G \subseteq S_4$)* |

---

## 3. The Prompt Used with the LLM (Task 2)

```text
I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain:
- a name;
- positive preconditions;
- negative preconditions;
- positive effects;
- negative effects.

An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied:
1. remove its negative effects from the state;
2. add its positive effects to the state.

Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also:
- detect when no plan exists;
- print the resulting sequence of actions;
- print the states reached after each action.

Explain the implementation and identify any assumptions you make.
Run the generated program on the warehouse problem.
```

---

## 4. Generated Python Program (Task 2 & 3)

> [!NOTE]
> The Python code below was generated with the assistance of an LLM and is saved in [`planner.py`](file:///d:/Acads/AI/Logic/planner.py).

```python
import copy
from collections import deque

class Action:
    def __init__(self, name, pos_preconds, neg_preconds, pos_effects, neg_effects):
        self.name = name
        self.pos_preconds = set(pos_preconds)
        self.neg_preconds = set(neg_preconds)
        self.pos_effects = set(pos_effects)
        self.neg_effects = set(neg_effects)

    def is_applicable(self, state):
        return self.pos_preconds.issubset(state) and len(self.neg_preconds.intersection(state)) == 0

    def apply(self, state):
        new_state = set(state)
        new_state.difference_update(self.neg_effects)
        new_state.update(self.pos_effects)
        return new_state

def bfs_plan(initial_state, goal_state, actions):
    queue = deque([(initial_state, [])])
    visited = set([frozenset(initial_state)])
    path_states = {frozenset(initial_state): [initial_state]}
    
    while queue:
        current_state, plan = queue.popleft()
        
        if goal_state.issubset(current_state):
            print("Plan found!")
            print(f"Initial State: {initial_state}")
            for i, action in enumerate(plan):
                print(f"Step {i+1}: {action.name}")
                print(f"  State reached: {path_states[frozenset(current_state)][i+1]}")
            return plan
            
        for action in actions:
            if action.is_applicable(current_state):
                next_state = action.apply(current_state)
                next_state_fs = frozenset(next_state)
                if next_state_fs not in visited:
                    visited.add(next_state_fs)
                    new_plan = plan + [action]
                    queue.append((next_state, new_plan))
                    
                    new_path_states = list(path_states[frozenset(current_state)])
                    new_path_states.append(next_state)
                    path_states[next_state_fs] = new_path_states
                    
    print("No plan found")
    return None
```

---

## 5. Test Results and Validation (Task 3)

### Test A: Solvable Problem
- **Initial State:** `{'At(Robot, A)', 'At(Package, A)'}`
- **Goal State:** `{'At(Package, C)'}`
- **Plan Found:** Yes
- **Resulting Plan:**
  1. `PickUp(Package, A)` $\rightarrow$ State: `{'At(Robot, A)', 'Holding(Package)'}`
  2. `Move(A, B)` $\rightarrow$ State: `{'At(Robot, B)', 'Holding(Package)'}`
  3. `Move(B, C)` $\rightarrow$ State: `{'At(Robot, C)', 'Holding(Package)'}`
  4. `Drop(Package, C)` $\rightarrow$ State: `{'At(Robot, C)', 'At(Package, C)'}`
- **Plan Validity:** **Valid.** Every action precondition is satisfied prior to execution, and the final state satisfies the goal.

### Test B: Impossible Problem (PickUp Action Removed)
- **Initial State:** `{'At(Robot, A)', 'At(Package, A)'}`
- **Goal State:** `{'At(Package, C)'}`
- **Plan Found:** No (`No plan found` reported)
- **Plan Validity:** **Valid behaviour.** The planner correctly reports that no plan exists rather than inventing illegal actions.

### Test C: Irrelevant Actions Added
- **Initial State:** `{'At(Robot, A)', 'At(Package, A)'}`
- **Goal State:** `{'At(Package, C)'}`
- **Actions Added:** `MoveAround(A)`, `MoveAround(B)`, `MoveAround(C)` (robot moves without carrying package)
- **Plan Found:** Yes (Optimal 4-step plan returned)
- **Plan Validity:** **Valid.** The planner avoids loops with irrelevant actions because BFS finds the shortest path to goal.

---

## 6. Task 4: Logic and Search Analysis

### Architecture Mapping
- **Logical Reasoning:** Determines action applicability ($S \models \text{Preconditions}(a)$) and computes valid successor states ($S' = \text{Apply}(S, a)$).
- **Search (BFS):** Explores the graph of states level-by-level to locate a sequence of actions from $I$ to a state satisfying $G$.

### Flowchart Sequence Completion (Page 9)
```text
Current state
    ↓
Check action preconditions
    ↓
Are preconditions satisfied? (Applicable actions)
    ↓
Generate successor state
    ↓
Search over alternatives
    ↓
Goal satisfied?
```

### Explanation of Co-operation:
Logic acts as a **filter and transition rule engine** (determining *what actions are physically possible* in any given state), while Search acts as the **strategy engine** (determining *which valid branches to explore next* to reach the goal efficiently).

---

## 7. Task 5: LLM Self-Verification Analysis

### Question: Which should you trust more: (a) LLM's explanation or (b) independently executed state transitions?
- **Answer:** **(b) Independently executed state transitions.**
- **Reasoning:** An LLM generates natural language text probabilistically. While its explanations may sound convincing and syntactically sound, they can contain subtle hallucinations or invalid logic. Independent state transition execution operates deterministically on formal mathematical set operations, guaranteeing correctness.

> [!IMPORTANT]
> **Engineering Principle:** *A generated explanation is not the same as an independent verification.*

---

## 8. Section 5: Reflection Questions (PDF Page 10)

1. **Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**
   - *Answer:* Formally specifying preconditions and effects establishes an unambiguous mathematical domain model ($I, A, G$). This eliminates ambiguities inherent in natural language prompts and ensures the generated code correctly implements state transition rules (`is_applicable` and `apply`) without fabricating illegal preconditions or effects.

2. **Give an example of an error that could occur if the planner failed to check an action's preconditions.**
   - *Answer:* If the planner executes `Drop(Package, C)` without checking its preconditions (`At(Robot, C)` and `Holding(Package)`), the robot could "teleport" the package to location $C$ from anywhere, or drop a package it never picked up.

3. **Why is a plan that "looks reasonable" not necessarily a valid plan?**
   - *Answer:* A plan might sound plausible to human intuition or an LLM (e.g., moving directly from $A$ to $C$), but fail because it violates underlying logical constraints, such as topological disconnects (no direct path between $A$ and $C$) or missing prerequisite steps.

4. **What did the LLM contribute to the implementation?**
   - *Answer:* The LLM contributed the core Python structure, data modeling (`Action` class), queue-driven BFS loop implementation, state tracing logic, and test harness routines.

5. **What did you have to verify independently?**
   - *Answer:* We independently verified that:
     1. Precondition checking correctly evaluates positive and negative conditions.
     2. Set updates in `apply()` mutate states correctly.
     3. BFS visited-set management prevents infinite loops.
     4. Test outputs strictly match domain transition rules.

6. **In this laboratory, where is logical reasoning being used?**
   - *Answer:* Logical reasoning is used in:
     - Evaluating $S \models \text{Preconditions}(a)$ to check action applicability.
     - Computing state transitions $S' = (S \setminus \text{Effects}^-(a)) \cup \text{Effects}^+(a)$.
     - Prolog Backward Chaining during formal plan verification.

7. **How is planning related to the search algorithms studied in the previous module?**
   - *Answer:* Planning is a specialized form of state-space search. States correspond to sets of logical facts, actions correspond to state transitions constrained by preconditions/effects, and goal checking evaluates whether goal propositions form a subset of the current state.

---

## 9. Task 6: Prolog as a Plan Verifier (`planner.pl`)

Prolog code saved in [`planner.pl`](file:///d:/Acads/AI/Logic/planner.pl):

```prolog
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

can_move(X, Y) :-
    connected(X, Y).
```

### Query Execution & Results
- `?- can_move(a, b).` $\rightarrow$ **`true.`**
- `?- can_move(a, c).` $\rightarrow$ **`false.`**

### Task 6 Question Answers
- **(a) Why does Prolog return `true` for `can_move(a,b)`?**
  Prolog matches `can_move(a,b)` to rule `can_move(X,Y) :- connected(X,Y)` with substitution `{X=a, Y=b}`. It checks subgoal `connected(a,b)`, which directly matches a fact in the knowledge base.
- **(b) Why does it not establish `can_move(a,c)`?**
  Prolog searches for `connected(a,c)` or an applicable rule. Since `connected(a,c)` is not in the knowledge base and no transitive rule is defined, Prolog applies the Closed World Assumption (CWA) and returns `false`.
- **(c) What is the relationship between the Prolog rule `can_move(X,Y)` and the logical implication $\text{Connected}(X,Y) \to \text{CanMove}(X,Y)$?**
  In Prolog syntax, `Head :- Body` denotes logical implication in reverse direction ($\text{Body} \implies \text{Head}$). The rule represents $\forall X, Y (\text{Connected}(X,Y) \implies \text{CanMove}(X,Y))$.

---

## 10. Task 7: Using Prolog to Check a Proposed Plan (`planner.pl`)

Extended Prolog rules in [`planner.pl`](file:///d:/Acads/AI/Logic/planner.pl):

```prolog
valid_move(X, Y) :-
    connected(X, Y).

valid_plan([]).
valid_plan([move(X,Y)|Rest]) :-
    valid_move(X, Y),
    valid_plan(Rest).
```

### Query Execution Results
- `?- valid_move(a, b).` $\rightarrow$ **`true.`**
- `?- valid_move(b, c).` $\rightarrow$ **`true.`**
- `?- valid_move(a, c).` $\rightarrow$ **`false.`**
- `?- valid_plan([move(a,b), move(b,c)]).` $\rightarrow$ **`true.`**

### Challenge Analysis (Action `Move(a,c)`)
- Query: `?- valid_move(a, c).` $\rightarrow$ **`false.`**
- **Conclusion:** **ACTION REJECTED.** The shortcut `Move(a,c)` is rejected because location $a$ is not directly connected to location $c$ in the warehouse KB.
- **Architecture Takeaway:** Illustrates the **Generate $\rightarrow$ Independent Verification** paradigm, where a search/AI agent generates candidate steps while Prolog independently enforces strict domain constraints.

---

## 11. Task 8: Connect Prolog to Logical Reasoning (`planner.pl`)

Prolog program added to [`planner.pl`](file:///d:/Acads/AI/Logic/planner.pl):

```prolog
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
```

### Query Execution Result
- `?- reduce_speed.` $\rightarrow$ **`true.`**

### Derivation Explanation
1. Goal `reduce_speed` matches rule `reduce_speed :- slippery`, creating subgoal `slippery`.
2. Goal `slippery` matches rule `slippery :- wet_road`, creating subgoal `wet_road`.
3. Goal `wet_road` matches ground fact `wet_road.` unconditionally.
4. Backward chaining resolves all subgoals to `true`.

### Required Implication Sequence Format
**Fact $\implies$ Rule $\implies$ Rule $\implies$ Conclusion**

$$\text{wet\_road} \implies (\text{wet\_road} \rightarrow \text{slippery}) \implies (\text{slippery} \rightarrow \text{reduce\_speed}) \implies \text{reduce\_speed}$$

---

## 12. Section 7.2: Prolog Reflection Questions (PDF Page 14)

1. **What is the difference between a Prolog fact and a Prolog rule?**
   - *Answer:* A fact is an unconditional assertion of truth (e.g., `wet_road.`), while a rule is a conditional assertion whose truth depends on one or more body subgoals being satisfied (e.g., `slippery :- wet_road.`).

2. **How does a Prolog query correspond to asking whether something follows from a knowledge base?**
   - *Answer:* A query submits a logical proposition to Prolog's inference engine. Prolog attempts to construct a formal proof via SLD resolution and unification to determine whether the proposition is logically entailed by the KB.

3. **Why might it be useful to use a Prolog program to verify a plan generated by a Python program?**
   - *Answer:* Using Prolog decouples plan generation from plan verification. Imperative Python code might have procedural bugs, whereas declarative Prolog logic provides an independent, mathematically rigorous validation layer.

4. **What advantage does an independent verifier provide when the original plan was generated with the help of an LLM?**
   - *Answer:* An independent verifier guarantees deterministic compliance with domain rules, catching hallucinations, illegal shortcut moves, or invalid preconditions that an LLM might mistakenly produce.
