# Laboratory Report – Logical Reasoning for Planning

## Task 0: Understand the Planning Problem

### (a) Initial state $I$
$$I = \{\text{At}(\text{Robot}, A), \text{At}(\text{Package}, A)\}$$

### (b) Goal $G$
$$G = \{\text{At}(\text{Package}, C)\}$$

### (c) Actions available to the robot
- $\text{Move}(X, Y)$
- $\text{PickUp}(\text{Package}, X)$
- $\text{Drop}(\text{Package}, X)$

### (d) Action Preconditions and Effects

| Action | Positive Preconditions | Negative Preconditions | Positive Effects | Negative Effects |
| :--- | :--- | :--- | :--- | :--- |
| `Move(X, Y)` | `At(Robot, X)` | $\emptyset$ | `At(Robot, Y)` | `At(Robot, X)` |
| `PickUp(Package, X)` | `At(Robot, X)`, `At(Package, X)` | $\emptyset$ | `Holding(Package)` | `At(Package, X)` |
| `Drop(Package, X)` | `At(Robot, X)`, `Holding(Package)` | $\emptyset$ | `At(Package, X)` | `Holding(Package)` |

### Task 0 Applicability Questions:
- **Is `PickUp(Package, A)` applicable initially?**
  **Yes.** Its preconditions are `At(Robot, A)` and `At(Package, A)`, both of which are present in the initial state $I$.
- **What about `Drop(Package, C)`?**
  **No.** Its preconditions are `At(Robot, C)` and `Holding(Package)`, neither of which is present in initial state $I$.

---

## Task 1: Construct a Plan by Hand

| State | Facts in Knowledge Base | Action |
| :--- | :--- | :--- |
| **$S_0$** | `At(Robot, A)`, `At(Package, A)` | `PickUp(Package, A)` |
| **$S_1$** | `At(Robot, A)`, `Holding(Package)` | `Move(A, B)` |
| **$S_2$** | `At(Robot, B)`, `Holding(Package)` | `Move(B, C)` |
| **$S_3$** | `At(Robot, C)`, `Holding(Package)` | `Drop(Package, C)` |
| **$S_4$** | `At(Robot, C)`, `At(Package, C)` | *Goal Achieved ($G \subseteq S_4$)* |

---

## Task 2 & 3: Implement Planner and Test

Implementation is contained in [`planner.py`](file:///d:/Acads/AI/Logic/planner.py).

### Test Results Summary:
- **Test A (Solvable Problem):** Plan found: `PickUp(Package, A)` $\rightarrow$ `Move(A, B)` $\rightarrow$ `Move(B, C)` $\rightarrow$ `Drop(Package, C)`. All actions are verified valid.
- **Test B (Impossible Problem):** PickUp action removed. Result: `No plan found`. Correctly detects absence of valid plan.
- **Test C (Irrelevant Actions):** Additional MoveAround actions added. Result: Planner finds the optimal 4-step plan without getting trapped in loops.

---

## Task 4: Logic and Search

### Completed Description Diagram (PDF Page 9):
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

### Explanation of Logical Reasoning and Search Interaction:
- **Logical Reasoning:** Evaluates whether an action is applicable ($S \models \text{Preconditions}(a)$) and computes the exact successor state ($S' = \text{Apply}(S, a)$).
- **Search (BFS):** Explores alternative action sequences level-by-level to guarantee finding the shortest path to the goal state.

---

## Task 5: Verify Plan (LLM Self-Verification)

- **Question:** Which should you trust more: (a) the LLM's explanation, or (b) the independently executed state transitions?
- **Answer:** **(b) The independently executed state transitions.**
- **Reason:** An LLM's natural language output can sound convincing while hiding logical hallucinations or illegal state jumps. Independent execution strictly enforces deterministic propositional state transition logic.

---

## Section 5: Reflection Questions (PDF Page 10)

1. **Why is it useful to specify action preconditions and effects before asking an LLM to write the planner?**
   Defining formal preconditions and effects creates a precise domain model ($I, A, G$). This eliminates natural language ambiguity and gives the LLM explicit rules for state validity and state transitions.

2. **Give an example of an error that could occur if the planner failed to check an action's preconditions.**
   The planner might execute `Drop(Package, C)` when the robot is at location $A$ without holding the package, causing illegal package teleportation.

3. **Why is a plan that "looks reasonable" not necessarily a valid plan?**
   A plan may appear plausible on the surface (e.g., moving directly from $A$ to $C$), but violate hidden domain rules such as non-existent physical connections between locations.

4. **What did the LLM contribute to the implementation?**
   The LLM generated the Python program skeleton (`Action` class, state set operations, BFS queue processing, and test case execution routines).

5. **What did you have to verify independently?**
   We verified that set operations correctly enforce positive/negative preconditions and effects, BFS search terminates properly, and test outputs match valid warehouse state transitions.

6. **In this laboratory, where is logical reasoning being used?**
   Logical reasoning is used in precondition checking ($S \models \text{Preconditions}(a)$), state update application ($S' = \text{Apply}(S, a)$), and Prolog rule resolution.

7. **How is planning related to the search algorithms studied in the previous module?**
   Planning is state-space search where nodes are logical state descriptions and edges are action applications constrained by logical preconditions.

---

## Task 6: Prolog as a Plan Verifier (`planner.pl`)

Prolog facts and rule in [`planner.pl`](file:///d:/Acads/AI/Logic/planner.pl):
```prolog
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

can_move(X, Y) :-
    connected(X, Y).
```

### Query Executions and Results:
- `?- can_move(a, b).` $\rightarrow$ **`true.`**
- `?- can_move(a, c).` $\rightarrow$ **`false.`**

### Questions:
(a) **Why does Prolog return true for `can_move(a,b)`?**
Prolog unifies `can_move(a,b)` with `can_move(X,Y) :- connected(X,Y)` under `{X=a, Y=b}`, and finds `connected(a,b).` as a true fact in the KB.

(b) **Why does it not establish `can_move(a,c)`?**
`connected(a,c)` is not in the knowledge base and no reachability rule exists. By the Closed World Assumption (CWA), Prolog returns `false`.

(c) **Relationship to logical implication $\text{Connected}(X,Y) \to \text{CanMove}(X,Y)$:**
In Prolog, `Head :- Body` represents reverse logical implication ($\text{Body} \implies \text{Head}$). Thus `can_move(X,Y) :- connected(X,Y)` represents $\forall X, Y (\text{Connected}(X,Y) \implies \text{CanMove}(X,Y))$.

---

## Task 7: Using Prolog to Check a Proposed Plan (`planner.pl`)

Extended Prolog rules in [`planner.pl`](file:///d:/Acads/AI/Logic/planner.pl):
```prolog
valid_move(X, Y) :-
    connected(X, Y).

valid_plan([]).
valid_plan([move(X,Y)|Rest]) :-
    valid_move(X, Y),
    valid_plan(Rest).
```

### Plan Query Results:
- `?- valid_move(a, b).` $\rightarrow$ **`true.`**
- `?- valid_move(b, c).` $\rightarrow$ **`true.`**
- `?- valid_move(a, c).` $\rightarrow$ **`false.`**
- `?- valid_plan([move(a,b), move(b,c)]).` $\rightarrow$ **`true.`**

### Challenge Analysis:
- Proposed Action: `Move(a, c)`
- Query: `?- valid_move(a, c).` $\rightarrow$ **`false.`**
- **Result:** **ACTION REJECTED.**
- **Takeaway:** Demonstrates **Generate $\rightarrow$ Independent Verification**. An AI component generates candidate actions, and Prolog independently verifies them against domain facts.

---

## Task 8: Connect Prolog to Logical Reasoning (`planner.pl`)

Prolog program added to [`planner.pl`](file:///d:/Acads/AI/Logic/planner.pl):
```prolog
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
```

### Query Execution & Explanation:
- `?- reduce_speed.` $\rightarrow$ **`true.`**
- **Explanation:** Prolog resolves `reduce_speed` to subgoal `slippery`, which resolves to subgoal `wet_road`, which matches the ground fact `wet_road.`.

### Sequence of Logical Implications:
**Format:** `Fact => Rule => Rule => Conclusion`

$$\text{wet\_road} \implies (\text{wet\_road} \rightarrow \text{slippery}) \implies (\text{slippery} \rightarrow \text{reduce\_speed}) \implies \text{reduce\_speed}$$

---

## Section 7.2: Reflection Questions (PDF Page 14)

1. **What is the difference between a Prolog fact and a Prolog rule?**
   A fact asserts unconditional truth (`wet_road.`), while a rule asserts conditional truth dependent on subgoals (`slippery :- wet_road.`).

2. **How does a Prolog query correspond to asking whether something follows from a knowledge base?**
   A query asks Prolog to perform SLD resolution to check whether the proposition is logically entailed by the KB.

3. **Why might it be useful to use a Prolog program to verify a plan generated by a Python program?**
   It decouples plan generation from validation, ensuring imperative Python code bugs do not produce illegal state transitions.

4. **What advantage does an independent verifier provide when the original plan was generated with the help of an LLM?**
   An independent verifier provides deterministic validation, preventing LLM hallucinations or illegal moves from being executed.
