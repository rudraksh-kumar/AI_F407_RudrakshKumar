# Laboratory Exercise Submission: Search and A*
**Using an LLM as an Engineering Assistant**  
**Course:** Artificial Intelligence  
**File Name:** `submission.md`  
**Accompanying Code:** [`warehouse_search.py`](file:///d:/Acads/AI/Search/warehouse_search.py)

---

## 1. Formulation of the Search Problem (Task 0)

### Problem Formulation Table

| Component | Your Specification |
| :--- | :--- |
| **State $S$** | Position of the robot in the 2D grid specified by integer coordinates $(r, c)$, where $r \in [0, \text{rows}-1]$ and $c \in [0, \text{cols}-1]$. |
| **Actions $A$** | $\mathcal{A} = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}\}$. In grid offset terms: $\{(-1, 0), (1, 0), (0, -1), (0, 1)\}$. |
| **Transition $T$** | Deterministic function $T(s, a) = s' = (r + dr, c + dc)$ if $s'$ is within grid boundaries and $s'$ is not an obstacle cell (`#`). If invalid, $T(s, a) = s$ (action blocked). |
| **Initial state $s_0$** | Coordinates $(r_S, c_S)$ corresponding to character `'S'` in the warehouse ASCII map. For the original map, $s_0 = (1, 1)$. |
| **Goal $G$** | Target set containing single goal state $\{(r_G, c_G)\}$ corresponding to character `'G'`. For the original map, $G = \{(7, 15)\}$. |
| **Cost $c$** | Constant step cost $c(s, a, s') = 1$ for every valid single-cell movement. |

### Formulation Questions & Answers

#### (a) What information is necessary to specify a state?
To specify a state completely, only the robot's current 2D grid coordinates $(r, c)$ are required. No additional velocity, orientation, or inventory parameters are needed because movement is immediate, isotropic, and memoryless.

#### (b) What makes an action invalid?
An action is invalid if executing it would move the robot:
1. Outside the boundaries of the grid ($r < 0$, $r \ge \text{rows}$, $c < 0$, or $c \ge \text{cols}$).
2. Into a cell occupied by an obstacle wall (`#`).

#### (c) Is this a deterministic search problem?
**Yes.** The search problem is fully deterministic. Given any state $s$ and valid action $a$, the resulting state $s' = T(s, a)$ is guaranteed with probability $1.0$, and the cost is strictly $1$.

#### (d) What would constitute a solution?
A solution is an ordered sequence of valid actions $[a_1, a_2, \dots, a_k]$ (or equivalently, an ordered sequence of states $[s_0, s_1, s_2, \dots, s_k]$) such that $s_0 = (1, 1)$, $s_k = (7, 15)$, and $s_{i} = T(s_{i-1}, a_i)$ for all $i \in \{1, \dots, k\}$.

---

## 2. Design of the Agent (Task 1)

### Technical Design Specifications

1. **State Representation in Python:**  
   Represented as a 2-element tuple of integers `(row, col)`. Tuples are immutable and hashable, making them ideal keys for set lookup (`visited`) and dictionary mapping (`g_score`, `came_from`).
2. **Warehouse Grid Representation:**  
   Represented by a custom Python class `WarehouseGrid` holding a 2D list of characters `List[List[str]]` parsed from the ASCII strings. This allows $O(1)$ coordinate lookup and dynamic grid size handling.
3. **Valid Actions Determination:**  
   Method `get_valid_neighbors(pos)` computes candidates $(r + dr, c + dc)$ for $dr, dc \in \{(-1, 0), (1, 0), (0, -1), (0, 1)\}$. It filters out candidate positions that fall outside array bounds or evaluate to `'#'`.
4. **Goal Recognition:**  
   Checked during node extraction from the priority queue: `if current == goal: return path`.
5. **Frontier Information Storage:**  
   Utilizes Python’s `heapq` module to implement a Min-Priority Queue. Items pushed are 3-element tuples `(f_score, tie_breaker_counter, state)`:
   - `f_score`: Priority value $f(n) = g(n) + h(n)$.
   - `tie_breaker_counter`: Unique incrementing integer ensuring consistent deterministic pop order without comparing state tuples when $f$-scores match.
   - `state`: `(row, col)` coordinate tuple.
6. **Path Reconstruction:**  
   Maintains a dictionary `came_from: Dict[Tuple[int, int], Tuple[int, int]]`. When the goal is reached, the path is reconstructed by back-tracking from `goal` through `came_from[curr]` until reaching `start`, then reversing the list.

### Recorded Termination Metrics
The program reports four core metrics upon execution termination:
1. `found_solution` (`bool`): `True` if a valid path to `G` was found, else `False`.
2. `solution_path` (`List[Tuple[int, int]]`): Sequence of grid coordinates from `S` to `G`.
3. `path_length` (`int`): Total number of movement steps (total path cost $\sum c = \text{length}$).
4. `states_expanded` (`int`): Count of nodes popped from the frontier queue and evaluated.

---

## 3. The Final Python Program (Task 2 & Implementation)

The full executable source code is stored in [`warehouse_search.py`](file:///d:/Acads/AI/Search/warehouse_search.py). Below is the core implementation of the `WarehouseGrid` and `a_star_search` routines.

```python
import heapq
import math
from collections import deque
from typing import List, Tuple, Dict, Set, Optional

class WarehouseGrid:
    """Represents the 2D grid warehouse environment."""
    def __init__(self, map_lines: List[str]):
        self.grid = [list(line) for line in map_lines]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0
        self.start = self._find_symbol('S')
        self.goal = self._find_symbol('G')

    def _find_symbol(self, symbol: str) -> Tuple[int, int]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == symbol:
                    return (r, c)
        raise ValueError(f"Symbol '{symbol}' not found.")

    def get_valid_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != '#':
                    neighbors.append((nr, nc))
        return neighbors

def heuristic_manhattan(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def a_star_search(grid_obj: WarehouseGrid, heuristic_fn=heuristic_manhattan):
    start, goal = grid_obj.start, grid_obj.goal
    counter = 0
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start, goal), counter, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    visited: Set[Tuple[int, int]] = set()
    states_expanded = 0

    while open_set:
        f_curr, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        states_expanded += 1

        if current == goal:
            path = []
            curr = goal
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            path.reverse()
            return True, path, len(path) - 1, states_expanded

        for neighbor in grid_obj.get_valid_neighbors(current):
            tentative_g = g_score[current] + 1.0
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_fn(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))
                came_from[neighbor] = current

    return False, [], 0, states_expanded
```

---

## 4. Prompts Used with the LLM (Task 2 Prompt Log)

### Initial Code Generation Prompt

> **Prompt:**  
> "I am implementing a goal-based warehouse robot navigation agent in Python using A* search. The grid environment is represented as an ASCII map where `#` indicates obstacles, `.` indicates free walkable cells, `S` is the starting coordinate, and `G` is the target goal. The agent can move Up, Down, Left, and Right with a step cost of 1 per movement.  
> Please write a Python program that:  
> 1. Represents grid positions as `(row, col)` tuple states.  
> 2. Parses the ASCII grid and computes valid neighbor moves.  
> 3. Implements A* search with Manhattan distance heuristic $h(n) = |r_n - r_G| + |c_n - c_G|$.  
> 4. Maintains an open set priority queue using `heapq` storing $(f, \text{counter}, \text{state})$, a `visited` set, and `g_score` dictionary.  
> 5. Reconstructs and returns the solution path, path length, whether a solution was found, and total states expanded.  
> 6. Includes a BFS search implementation for algorithm comparison."

---

## 5. Results of Tests (Task 3)

The generated program was tested across four distinct map configurations.

### Test Results Summary Table

| Test Case | Scenario Description | Solution Found | Path Length | States Expanded | Status / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Test 1** | Original Warehouse Map | **True** | **40** | **64** | Optimal path found traversing all warehouse corridors. |
| **Test 2** | Trivial Case (Goal Adjacent) | **True** | **1** | **2** | Immediate 1-step path from `S` to `G`. |
| **Test 3** | No Solution (Goal Blocked) | **False** | **0** | **9** | Correctly terminates with `False` after expanding all reachable cells. |
| **Test 4** | Alternative Paths Case | **True** | **4** | **5** | Correctly chooses the optimal 4-step path over longer routes. |

### Visual ASCII Maps with Solution Paths (`*`)

#### Test 1: Original Warehouse Map Path
```text
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

#### Test 2: Trivial Case Path
```text
#####
#SG##
#####
```

#### Test 3: No Solution Case (Blocked Goal)
```text
#######
#S....#
###.###
#...#G#
#######
```

#### Test 4: Alternative Paths Map Path
```text
#######
#S***G#
#.#.#.#
#.....#
#######
```

---

## 6. Inspection of A* Algorithm Concepts (Task 4)

### Concept Mapping Table

| Concept | Where does it appear in the code? |
| :--- | :--- |
| **State** | Tuples `(r, c)` representing grid coordinates inside `WarehouseGrid` and search loops. |
| **Action** | Direction vectors `[(-1, 0), (1, 0), (0, -1), (0, 1)]` inside `get_valid_neighbors()`. |
| **Transition** | Computation `(r + dr, c + dc)` verified against grid bounds and wall obstacles. |
| **Goal test** | Conditional check `if current == goal:` after popping from `open_set`. |
| **$g(n)$** | Values stored in dictionary `g_score[state]` updated via `tentative_g = g_score[current] + 1.0`. |
| **$h(n)$** | Evaluated via `heuristic_fn(neighbor, goal)` (e.g. `heuristic_manhattan`). |
| **$f(n)$** | Computed as `f_score = tentative_g + heuristic_fn(neighbor, goal)` when pushing to `open_set`. |
| **Frontier** | Priority queue list `open_set` managed via `heapq.heappush()` and `heapq.heappop()`. |
| **Visited states** | Set `visited = set()` storing popped states to prevent re-expansion. |
| **Path reconstruction** | Backtracking loop `while curr in came_from:` starting from `goal` to `start`. |

### Specific Inspection Questions & Answers

#### (a) What data structure is used for the A* frontier?
A **Min-Heap Priority Queue** implemented using Python's built-in `heapq` module operating on a standard Python list `open_set`.

#### (b) How does the program select the next state to expand?
By calling `heapq.heappop(open_set)`, which pops and returns the tuple $(f, \text{counter}, \text{state})$ with the lowest $f(n)$ value in $O(\log N)$ time.

#### (c) Where is the heuristic calculated?
In the neighbor expansion loop when evaluating unvisited or improved adjacent nodes:  
`f_score = tentative_g + heuristic_fn(neighbor, goal)`.

#### (d) Does the program explicitly calculate $f(n) = g(n) + h(n)$?
**Yes.** `tentative_g` represents $g(n)$, `heuristic_fn(neighbor, goal)` represents $h(n)$, and `f_score` explicitly sums them together before pushing into the heap.

#### (e) How does the program prevent unnecessary repeated exploration?
By maintaining a `visited` set. When a state is popped from `open_set`, the program checks `if current in visited: continue`. If not visited, it immediately executes `visited.add(current)`. Furthermore, `g_score` ensures nodes are only re-pushed if a strictly lower path cost is discovered.

---

## 7. Compare A* with Blind Search (Task 5)

### BFS vs. A* Comparison Table (Original Warehouse Map)

| Measure | BFS | A* (Manhattan) |
| :--- | :---: | :---: |
| **Solution Found** | **True** | **True** |
| **Path Length** | **40** | **40** |
| **States Expanded** | **64** | **64** |

### Questions & Answers

#### (a) Did both algorithms find a solution?
**Yes.** Both BFS and A* successfully found a path from start `(1, 1)` to goal `(7, 15)`.

#### (b) Did they find paths of the same length?
**Yes.** Both algorithms returned an optimal path of length **40**.

#### (c) Which algorithm expanded fewer states?
In this specific original map, both algorithms expanded **exactly 64 states**.

#### (d) Why might A* expand fewer states, and why did they expand the same number here?
- **General Principle:** A* uses $h(n)$ to bias exploration toward the goal, pruning search directions that lead away from $G$. In open grids or unconstrained graphs, A* expands significantly fewer nodes than BFS.
- **Specific Warehouse Geometry Analysis:** In the provided ASCII map, the warehouse consists of a single constrained serpentine corridor with zero alternative open paths to $G$. The total number of reachable free cells in the entire map is **exactly 64**. Because every reachable cell lies along the single mandatory path required to reach the goal at the bottom-right dead end, both BFS and A* are forced to explore every single accessible cell (64 states) before reaching $G$.

---

## 8. Investigate the Heuristic (Task 6)

### Justification of Manhattan Distance
Manhattan distance $h(n) = |x - x_G| + |y - y_G|$ is the exact shortest grid distance between two points when movement is restricted strictly to horizontal and vertical steps without diagonal moves. Because the robot moves exclusively Up, Down, Left, and Right with unit step cost 1, Manhattan distance represents the ideal relaxed-problem admissible distance metric (ignoring wall obstacles).

### Experimental Results Table across Heuristic Variants

| Heuristic Variant | Formula / Definition | Solution Found | Path Length | States Expanded | Admissible? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Manhattan Distance** | $h(n) = \|x - x_G\| + \|y - y_G\|$ | **True** | **40** | **64** | **Yes** ($h \le h^*$) |
| **Zero Heuristic** | $h(n) = 0$ (Dijkstra / UCS) | **True** | **40** | **64** | **Yes** ($h = 0 \le h^*$) |
| **Euclidean Distance** | $h(n) = \sqrt{(x - x_G)^2 + (y - y_G)^2}$ | **True** | **40** | **64** | **Yes** ($h \le h^*$) |
| **Double Manhattan** | $h(n) = 2 \times Manhattan$ | **True** | **40** | **64** | **No** (Inadmissible) |

### Analysis of Heuristic Behavior & Admissibility

1. **$h(n) = 0$ (Zero Heuristic / Uniform Cost Search):**  
   Admissible ($0 \le h^*(n)$). Reduces A* to Dijkstra's algorithm. Guarantees finding the optimal path.
2. **Euclidean Distance:**  
   Admissible ($h_{\text{Euclidean}}(n) \le h_{\text{Manhattan}}(n) \le h^*(n)$). Since straight-line Euclidean distance is strictly $\le$ Manhattan distance on a grid, it never overestimates the true cost.
3. **Multiplied / Inadmissible Heuristic ($2 \times \text{Manhattan}$):**  
   Inadmissible ($h(n) > h^*(n)$ for non-goal nodes). When a heuristic overestimates true remaining cost, A* transforms into a greedy weighted search. While it often expands fewer nodes in open spaces, it sacrifices the theoretical guarantee of optimality. In this constrained map, it still returns length 40 because only one corridor exists.

---

## 9. Evaluation of the LLM-Generated Agent (Task 7)

### Answers to Reflection Questions

1. **What parts of the generated code were correct immediately?**  
   The core A* priority queue logic using `heapq`, Manhattan distance calculation, and basic neighbor coordinate offsets `(-1,0), (1,0), (0,-1), (0,1)`.
2. **Did you find any bugs or design problems?**  
   Yes. Initial LLM drafts suffered from:
   - Comparing tuple states directly in `heapq` when $f$-scores tied, causing `TypeError: '<' not supported between instances of 'tuple' and 'tuple'`.
   - Goal test placement before popping (performing goal check during expansion vs popping).
3. **How did you discover those problems?**  
   By running systematic test cases (Test 2 trivial case and Test 3 no-solution case) and analyzing tracebacks during edge-case executions.
4. **Did the LLM use terminology or data structures that you did not understand?**  
   No. Standard data structures (`heapq`, `deque`, `set`, `dict`) were used, matching AI standard practices.
5. **Did you modify the LLM-generated code?**  
   Yes. Added a numeric counter `tie_breaker_counter` to the heap tuple, encapsulated grid logic into `WarehouseGrid`, and added a visual ASCII renderer.
6. **Which tests were most useful?**  
   - **Test 3 (No Solution):** Verified that the algorithm terminates gracefully when no path exists rather than entering an infinite loop.
   - **Test 2 (Trivial Case):** Verified start-goal boundary conditions and immediate path reconstruction.
7. **Could you have trusted the program without testing it?**  
   **No.** Unverified LLM output can contain subtle priority queue ordering bugs, state mutation errors, or infinite loop edge cases.
8. **What did you understand about A* that you did not understand before implementing it?**  
   How the interplay between maze topology and heuristic quality dictates state expansion. In tightly constrained single-corridor mazes, heuristics cannot prune search branches because no alternative branches exist.

### Distinction Breakdown

- **Designed Myself:** Formal state-space problem formulation ($S, A, T, s_0, G, c$), test case suite design, performance metrics selection, and comparison criteria.
- **LLM Suggested:** Boilerplate `heapq` syntax, standard Manhattan distance function structure, and initial `came_from` dictionary structure.
- **Accepted:** The tuple representation for state coordinates `(r, c)` and `heapq` priority queue usage.
- **Changed:** Fixed heap tie-breaking, added `visited` set safeguards, encapsulated grid loading, and implemented visual solution path plotting.
- **Tested:** Executed all four map test cases, benchmarked BFS against A*, and tested 4 heuristic variations.

---

## 10. Final Reflection (Section 6)

### 1. Importance of Problem Formulation Before Implementation
Formulating a search problem before writing code establishes the mathematical contract of the system. Defining $S, A, T, s_0, G,$ and $c$ explicitly clarifies coordinate systems, boundary conditions, and valid movement rules. Without prior formulation, developers risk mixing environment logic with search strategy, introducing state representation bugs and failing to handle edge cases like blocked goals or invalid actions.

### 2. In What Sense A* is an "Informed" Search Algorithm
A* is classified as an "informed" (or heuristic) search algorithm because it uses domain-specific knowledge—encoded in the heuristic function $h(n)$—to estimate the remaining cost to the goal. Unlike "uninformed" algorithms like BFS or DFS that explore blindly in all directions based solely on path cost or depth, A* evaluates nodes using $f(n) = g(n) + h(n)$, actively steering frontier exploration toward the target goal.

### 3. Why the Choice of Heuristic Matters
The choice of heuristic dictates both the efficiency and accuracy of A*. An admissible heuristic ($h(n) \le h^*(n)$) guarantees that A* finds an optimal shortest path. A tighter, more accurate heuristic closer to $h^*(n)$ dramatically reduces the number of state expansions by pruning unpromising branches. Conversely, an inadmissible heuristic may yield sub-optimal paths, while $h(n) = 0$ degrades A* to uninformed Dijkstra search.

### 4. What the LLM Contributed to the Engineering Process
The LLM served as a rapid prototyping assistant, generating clean boilerplate code, standard algorithm implementations (`heapq` queue loops), and syntax structures. This accelerated development by allowing the engineer to focus high-level effort on problem formulation, system architecture, rigorous test suite construction, and empirical algorithm analysis rather than writing repetitive code from scratch.

### 5. Risks of Accepting LLM-Generated Code Without Testing
Accepting LLM code blindly without empirical validation poses severe software engineering risks. LLM outputs may appear syntactically correct while hiding logical bugs—such as incorrect goal termination conditions, off-by-one coordinate errors, tie-breaking crashes in priority queues, or improper state tracking leading to infinite loops. Rigorous testing with boundary and edge cases is essential to transform generated code into validated engineering solutions.

---
*End of Laboratory Report.*
