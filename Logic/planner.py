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
    
    # Track state sequence for printing
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

def run_test_A():
    print("--- Test A: Solvable Problem ---")
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    
    actions = [
        Action("Move(A, B)", {"At(Robot, A)"}, set(), {"At(Robot, B)"}, {"At(Robot, A)"}),
        Action("Move(B, A)", {"At(Robot, B)"}, set(), {"At(Robot, A)"}, {"At(Robot, B)"}),
        Action("Move(B, C)", {"At(Robot, B)"}, set(), {"At(Robot, C)"}, {"At(Robot, B)"}),
        Action("Move(C, B)", {"At(Robot, C)"}, set(), {"At(Robot, B)"}, {"At(Robot, C)"}),
        Action("PickUp(Package, A)", {"At(Robot, A)", "At(Package, A)"}, set(), {"Holding(Package)"}, {"At(Package, A)"}),
        Action("PickUp(Package, B)", {"At(Robot, B)", "At(Package, B)"}, set(), {"Holding(Package)"}, {"At(Package, B)"}),
        Action("PickUp(Package, C)", {"At(Robot, C)", "At(Package, C)"}, set(), {"Holding(Package)"}, {"At(Package, C)"}),
        Action("Drop(Package, A)", {"At(Robot, A)", "Holding(Package)"}, set(), {"At(Package, A)"}, {"Holding(Package)"}),
        Action("Drop(Package, B)", {"At(Robot, B)", "Holding(Package)"}, set(), {"At(Package, B)"}, {"Holding(Package)"}),
        Action("Drop(Package, C)", {"At(Robot, C)", "Holding(Package)"}, set(), {"At(Package, C)"}, {"Holding(Package)"}),
    ]
    
    bfs_plan(initial_state, goal, actions)

def run_test_B():
    print("\n--- Test B: Impossible Problem ---")
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    
    actions = [
        Action("Move(A, B)", {"At(Robot, A)"}, set(), {"At(Robot, B)"}, {"At(Robot, A)"}),
        Action("Move(B, A)", {"At(Robot, B)"}, set(), {"At(Robot, A)"}, {"At(Robot, B)"}),
        Action("Move(B, C)", {"At(Robot, B)"}, set(), {"At(Robot, C)"}, {"At(Robot, B)"}),
        Action("Move(C, B)", {"At(Robot, C)"}, set(), {"At(Robot, B)"}, {"At(Robot, C)"}),
        Action("Drop(Package, A)", {"At(Robot, A)", "Holding(Package)"}, set(), {"At(Package, A)"}, {"Holding(Package)"}),
        Action("Drop(Package, B)", {"At(Robot, B)", "Holding(Package)"}, set(), {"At(Package, B)"}, {"Holding(Package)"}),
        Action("Drop(Package, C)", {"At(Robot, C)", "Holding(Package)"}, set(), {"At(Package, C)"}, {"Holding(Package)"}),
    ]
    
    bfs_plan(initial_state, goal, actions)

def run_test_C():
    print("\n--- Test C: Irrelevant Actions ---")
    initial_state = {"At(Robot, A)", "At(Package, A)"}
    goal = {"At(Package, C)"}
    
    actions = [
        Action("Move(A, B)", {"At(Robot, A)"}, set(), {"At(Robot, B)"}, {"At(Robot, A)"}),
        Action("Move(B, A)", {"At(Robot, B)"}, set(), {"At(Robot, A)"}, {"At(Robot, B)"}),
        Action("Move(B, C)", {"At(Robot, B)"}, set(), {"At(Robot, C)"}, {"At(Robot, B)"}),
        Action("Move(C, B)", {"At(Robot, C)"}, set(), {"At(Robot, B)"}, {"At(Robot, C)"}),
        # Add irrelevant actions moving robot around
        Action("MoveAround(A)", {"At(Robot, A)"}, set(), set(), set()),
        Action("MoveAround(B)", {"At(Robot, B)"}, set(), set(), set()),
        Action("MoveAround(C)", {"At(Robot, C)"}, set(), set(), set()),
        Action("PickUp(Package, A)", {"At(Robot, A)", "At(Package, A)"}, set(), {"Holding(Package)"}, {"At(Package, A)"}),
        Action("PickUp(Package, B)", {"At(Robot, B)", "At(Package, B)"}, set(), {"Holding(Package)"}, {"At(Package, B)"}),
        Action("PickUp(Package, C)", {"At(Robot, C)", "At(Package, C)"}, set(), {"Holding(Package)"}, {"At(Package, C)"}),
        Action("Drop(Package, A)", {"At(Robot, A)", "Holding(Package)"}, set(), {"At(Package, A)"}, {"Holding(Package)"}),
        Action("Drop(Package, B)", {"At(Robot, B)", "Holding(Package)"}, set(), {"At(Package, B)"}, {"Holding(Package)"}),
        Action("Drop(Package, C)", {"At(Robot, C)", "Holding(Package)"}, set(), {"At(Package, C)"}, {"Holding(Package)"}),
    ]
    
    bfs_plan(initial_state, goal, actions)

if __name__ == "__main__":
    run_test_A()
    run_test_B()
    run_test_C()
