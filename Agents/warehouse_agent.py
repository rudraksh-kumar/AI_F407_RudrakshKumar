"""
Warehouse Navigation Goal-Based Agent
======================================
AI Laboratory Exercise: Goal-Based Intelligent Agent implementation for autonomous 
warehouse package navigation using Breadth-First Search (BFS) and A* Search algorithms.

Author: AI Lab Team
Date: 2026-09-30
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Dict, Set


# Default Warehouse Map from Laboratory Specifications
DEFAULT_GRID_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]


class WarehouseEnvironment:
    """
    Represents the warehouse environment as a 2D Grid.
    
    Grid Key:
      'S': Starting position of the autonomous vehicle
      'G': Goal / Dispatch area position
      '#': Shelf / Obstacle (impassable)
      '.': Free navigation space
    """
    def __init__(self, grid_str_list: List[str]):
        self.raw_grid = [list(row) for row in grid_str_list]
        self.height = len(self.raw_grid)
        self.width = len(self.raw_grid[0]) if self.height > 0 else 0
        self.start_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        
        self._parse_environment()

    def _parse_environment(self) -> None:
        """Locates start ('S') and goal ('G') coordinates within the grid."""
        for r in range(self.height):
            for c in range(self.width):
                cell = self.raw_grid[r][c]
                if cell == 'S':
                    self.start_pos = (r, c)
                elif cell == 'G':
                    self.goal_pos = (r, c)

        if not self.start_pos:
            raise ValueError("Start position 'S' not found in environment grid.")
        if not self.goal_pos:
            raise ValueError("Goal position 'G' not found in environment grid.")

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Checks if position is within bounds and not an obstacle ('#')."""
        r, c = pos
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.raw_grid[r][c] != '#'
        return False

    def get_successors(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Returns valid actions and resulting neighbor states.
        
        Actions:
          'UP'    : (-1, 0)
          'DOWN'  : (1, 0)
          'LEFT'  : (0, -1)
          'RIGHT' : (0, 1)
        """
        r, c = pos
        moves = [
            ('UP', (r - 1, c)),
            ('DOWN', (r + 1, c)),
            ('LEFT', (r, c - 1)),
            ('RIGHT', (r, c + 1))
        ]
        valid_moves = []
        for action, new_pos in moves:
            if self.is_valid_position(new_pos):
                valid_moves.append((action, new_pos))
        return valid_moves

    def render_path(self, path: List[Tuple[int, int]]) -> str:
        """Visualizes grid with path marked using '*' overlay."""
        grid_copy = [row[:] for row in self.raw_grid]
        path_set = set(path)
        
        for r, c in path:
            if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                grid_copy[r][c] = '*'
                
        rendered = []
        for row in grid_copy:
            rendered.append("".join(row))
        return "\n".join(rendered)


class GoalBasedAgent:
    """
    Goal-Based Intelligent Agent for Autonomous Warehouse Navigation.
    
    Attributes:
        env (WarehouseEnvironment): The perceived environment model.
        state (Tuple[int, int]): Current coordinates (row, col) of the agent.
        goal (Tuple[int, int]): Target coordinates (row, col).
    """

    def __init__(self, env: WarehouseEnvironment):
        self.env = env
        self.state = env.start_pos
        self.goal = env.goal_pos

    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        """Goal test predicate."""
        return state == self.goal

    def solve_bfs(self) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[str]], int]:
        """
        Decision-Making Component using Breadth-First Search (BFS).
        
        Guarantees shortest path (minimum number of moves) in an unweighted grid.
        Returns:
            (path_coords, action_sequence, nodes_expanded_count)
        """
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        # Queue storing tuple of: (current_state, path_so_far, actions_so_far)
        queue = deque([(start, [start], [])])
        visited: Set[Tuple[int, int]] = {start}
        nodes_expanded = 0

        while queue:
            current_pos, path, actions = queue.popleft()
            nodes_expanded += 1

            if current_pos == goal:
                return path, actions, nodes_expanded

            for action, neighbor in self.env.get_successors(current_pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], actions + [action]))

        return None, None, nodes_expanded

    def solve_astar(self) -> Tuple[Optional[List[Tuple[int, int]]], Optional[List[str]], int]:
        """
        Decision-Making Component using A* Search algorithm with Manhattan Distance heuristic.
        
        f(n) = g(n) + h(n)
          g(n): Exact cost from start to state n
          h(n): Manhattan distance from state n to goal
        Returns:
            (path_coords, action_sequence, nodes_expanded_count)
        """
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Priority Queue stores: (f_score, g_score, current_pos, path, actions)
        pq = []
        start_h = manhattan_distance(start, goal)
        heapq.heappush(pq, (start_h, 0, start, [start], []))
        
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        nodes_expanded = 0

        while pq:
            f, g, current_pos, path, actions = heapq.heappop(pq)
            nodes_expanded += 1

            if current_pos == goal:
                return path, actions, nodes_expanded

            for action, neighbor in self.env.get_successors(current_pos):
                tentative_g = g + 1
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + manhattan_distance(neighbor, goal)
                    heapq.heappush(pq, (f_score, tentative_g, neighbor, path + [neighbor], actions + [action]))

        return None, None, nodes_expanded


def explain_algorithm_choice() -> str:
    """Provides architectural explanation of the search algorithm choice."""
    return (
        "======================================================================\n"
        "SEARCH ALGORITHM SELECTION & JUSTIFICATION\n"
        "======================================================================\n"
        "Primary Algorithm Chosen: Breadth-First Search (BFS) / A* Search\n\n"
        "Why BFS is Appropriate:\n"
        "  1. Uniform Step Costs: Every move (Up, Down, Left, Right) has equal cost (1 unit).\n"
        "  2. Optimality Guarantee: BFS is guaranteed to find the shortest collision-free\n"
        "     path (fewest grid moves) in an unweighted search space.\n"
        "  3. Completeness: Since the grid state space is finite, BFS is complete\n"
        "     and will always terminate with either the optimal path or proof that\n"
        "     no path exists.\n\n"
        "Why A* Search with Manhattan Heuristic is also ideal:\n"
        "  1. Informed Search: Uses h(n) = |r1 - r2| + |c1 - c2| to guide search directly\n"
        "     towards the target dispatch area (G).\n"
        "  2. Efficiency: Expands fewer nodes than unguided BFS on larger warehouse maps\n"
        "     while retaining strict path optimality.\n"
        "======================================================================\n"
    )


def run_demonstration():
    """Runs full warehouse navigation demonstration and prints detailed logs."""
    print("=" * 70)
    print("AUTONOMOUS WAREHOUSE NAVIGATION - GOAL-BASED AGENT DEMONSTRATION")
    print("=" * 70)

    # Initialize Environment & Agent
    env = WarehouseEnvironment(DEFAULT_GRID_MAP)
    agent = GoalBasedAgent(env)

    print("\n[1] Initial Warehouse Map:")
    print("\n".join(DEFAULT_GRID_MAP))
    print(f"\nStart Position (S): {env.start_pos}")
    print(f"Goal Position  (G): {env.goal_pos}\n")

    # Algorithm Explanation
    print(explain_algorithm_choice())

    # Execute BFS Search
    print("\n[2] Executing Breadth-First Search (BFS)...")
    path_bfs, actions_bfs, nodes_bfs = agent.solve_bfs()

    if path_bfs:
        print(f"\nSUCCESS: Optimal Path Found using BFS!")
        print(f"Total Steps (Path Length): {len(actions_bfs)} moves")
        print(f"Total States Expanded: {nodes_bfs} nodes")
        print(f"\nAction Sequence:\n -> {' -> '.join(actions_bfs)}")
        print(f"\nCoordinate Path Sequence:\n{path_bfs}")
        
        print("\n[3] Visualized Path Map (* indicates traversed path):")
        print(env.render_path(path_bfs))
    else:
        print("\nFAILURE: No collision-free path exists from Start to Goal.")

    # Execute A* Search Comparison
    print("\n" + "-" * 70)
    print("[4] Comparison: Executing A* Search (Manhattan Distance Heuristic)...")
    path_astar, actions_astar, nodes_astar = agent.solve_astar()

    if path_astar:
        print(f"SUCCESS: Optimal Path Found using A*!")
        print(f"Total Steps (Path Length): {len(actions_astar)} moves")
        print(f"Total States Expanded: {nodes_astar} nodes")
        print(f"\nBFS Expanded Nodes: {nodes_bfs} | A* Expanded Nodes: {nodes_astar}")


if __name__ == "__main__":
    run_demonstration()
