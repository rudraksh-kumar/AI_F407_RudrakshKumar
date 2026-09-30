import heapq
import math
from collections import deque
from typing import List, Tuple, Dict, Set

ORIGINAL_MAP = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################"
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####"
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######"
]

ALTERNATIVE_PATHS_MAP = [
    "#######",
    "#S...G#",
    "#.#.#.#",
    "#.....#",
    "#######"
]


class WarehouseGrid:
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

    def render_with_path(self, path: List[Tuple[int, int]]) -> str:
        grid_copy = [list(line) for line in ["".join(row) for row in self.grid]]
        for r, c in set(path):
            if grid_copy[r][c] not in ('S', 'G'):
                grid_copy[r][c] = '*'
        return "\n".join("".join(row) for row in grid_copy)


def heuristic_manhattan(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def heuristic_zero(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return 0.0

def heuristic_euclidean(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def heuristic_double_manhattan(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return 2.0 * (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))


def a_star_search(grid_obj: WarehouseGrid, heuristic_fn=heuristic_manhattan) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    start, goal = grid_obj.start, grid_obj.goal
    counter = 0
    open_set = []
    heapq.heappush(open_set, (heuristic_fn(start, goal), counter, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    visited: Set[Tuple[int, int]] = set()
    states_expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)

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


def bfs_search(grid_obj: WarehouseGrid) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    start, goal = grid_obj.start, grid_obj.goal
    queue = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    states_expanded = 0

    while queue:
        current = queue.popleft()
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
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return False, [], 0, states_expanded


def main():
    print("=" * 60)
    print("TESTING PROGRAM ON MAPS")
    print("=" * 60)

    tests = [
        ("Test 1: Original Map", ORIGINAL_MAP),
        ("Test 2: Trivial Case", TRIVIAL_MAP),
        ("Test 3: No Solution Case", NO_SOLUTION_MAP),
        ("Test 4: Alternative Paths Case", ALTERNATIVE_PATHS_MAP),
    ]

    for name, map_data in tests:
        grid = WarehouseGrid(map_data)
        found, path, length, expanded = a_star_search(grid, heuristic_manhattan)
        print(f"\n--- {name} ---")
        print(f"Solution Found : {found}")
        print(f"Path Length    : {length}")
        print(f"States Expanded: {expanded}")
        print(f"Path           : {path}")
        if found:
            print("Visual Map with Path ('*'):")
            for line in grid.render_with_path(path).split("\n"):
                print(f"  {line}")

    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON: BFS vs A*")
    print("=" * 60)

    grid_orig = WarehouseGrid(ORIGINAL_MAP)
    bfs_f, bfs_p, bfs_l, bfs_e = bfs_search(grid_orig)
    astar_f, astar_p, astar_l, astar_e = a_star_search(grid_orig, heuristic_manhattan)

    print(f"{'Algorithm':<10} | {'Found':<6} | {'Length':<6} | {'Expanded':<8}")
    print("-" * 40)
    print(f"{'BFS':<10} | {str(bfs_f):<6} | {bfs_l:<6} | {bfs_e:<8}")
    print(f"{'A*':<10} | {str(astar_f):<6} | {astar_l:<6} | {astar_e:<8}")

    print("\n" + "=" * 60)
    print("HEURISTIC FUNCTION INVESTIGATION")
    print("=" * 60)

    h_variants = [
        ("Manhattan Distance", heuristic_manhattan),
        ("Zero Heuristic (h=0)", heuristic_zero),
        ("Euclidean Distance", heuristic_euclidean),
        ("Double Manhattan (2*h)", heuristic_double_manhattan)
    ]

    print(f"{'Heuristic':<25} | {'Found':<6} | {'Length':<6} | {'Expanded':<8}")
    print("-" * 55)
    for name, h_func in h_variants:
        f, p, l, e = a_star_search(grid_orig, h_func)
        print(f"{name:<25} | {str(f):<6} | {l:<6} | {e:<8}")


if __name__ == "__main__":
    main()
