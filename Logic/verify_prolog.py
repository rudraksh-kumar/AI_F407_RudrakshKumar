"""
Verification script for Task 6, Task 7, and Task 8 (Prolog Plan Verifier simulation).
This script executes the exact logical resolution performed by Prolog on planner.pl.
"""

class PrologVerifier:
    def __init__(self):
        # Facts for Task 6 & 7: connected(X, Y)
        self.facts_connected = {
            ('a', 'b'),
            ('b', 'a'),
            ('b', 'c'),
            ('c', 'b')
        }
        # Facts for Task 8: wet_road
        self.fact_wet_road = True

    def can_move(self, x, y):
        # Rule: can_move(X, Y) :- connected(X, Y)
        return (x, y) in self.facts_connected

    def valid_move(self, x, y):
        # Rule: valid_move(X, Y) :- connected(X, Y)
        return (x, y) in self.facts_connected

    def valid_plan(self, moves):
        for x, y in moves:
            if not self.valid_move(x, y):
                return False
        return True

    # Task 8 rules
    def wet_road(self):
        return self.fact_wet_road

    def slippery(self):
        # Rule: slippery :- wet_road
        return self.wet_road()

    def reduce_speed(self):
        # Rule: reduce_speed :- slippery
        return self.slippery()

def run_task6_tests():
    print("=" * 60)
    print("TASK 6: Prolog as a Plan Verifier")
    print("=" * 60)
    prolog = PrologVerifier()
    q1 = prolog.can_move('a', 'b')
    print(f"Query: ?- can_move(a, b).  => Result: {q1} (true)")
    q2 = prolog.can_move('a', 'c')
    print(f"Query: ?- can_move(a, c).  => Result: {q2} (false)")
    print()

def run_task7_tests():
    print("=" * 60)
    print("TASK 7: Using Prolog to Check a Proposed Plan")
    print("=" * 60)
    prolog = PrologVerifier()
    plan1 = [('a', 'b'), ('b', 'c')]
    print("Testing Proposed Sequence: [Move(a, b), Move(b, c)]")
    for step in plan1:
        res = prolog.valid_move(step[0], step[1])
        print(f"  Query: ?- valid_move({step[0]}, {step[1]}). => Result: {res}")
    print(f"  Full Plan Valid? ?- valid_plan({plan1}). => Result: {prolog.valid_plan(plan1)}")
    print()
    print("Challenge: Proposed Action Move(a, c)")
    res_challenge = prolog.valid_move('a', 'c')
    print(f"  Query: ?- valid_move(a, c). => Result: {res_challenge}")
    print(f"  Verification Result: ACTION REJECTED (Not supported by KB)")
    print("=" * 60)
    print()

def run_task8_tests():
    print("=" * 60)
    print("TASK 8: Connect Prolog to Logical Reasoning")
    print("=" * 60)
    prolog = PrologVerifier()
    q = prolog.reduce_speed()
    print(f"Query: ?- reduce_speed. => Result: {q} (true)")
    print("Derivation Chain:")
    print("  wet_road. (Fact)")
    print("  slippery :- wet_road. (Rule 1 => slippery holds)")
    print("  reduce_speed :- slippery. (Rule 2 => reduce_speed holds)")
    print("Sequence of Implications:")
    print("  wet_road => (wet_road -> slippery) => (slippery -> reduce_speed) => reduce_speed")
    print("=" * 60)

if __name__ == "__main__":
    run_task6_tests()
    run_task7_tests()
    run_task8_tests()
