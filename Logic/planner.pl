% ====================================================================
% Prolog Planner Verifier & Logical Reasoning (planner.pl)
% Warehouse Knowledge Base and Plan Verification Rules
% Tasks 6, 7 & 8 - Logical Reasoning for Planning
% ====================================================================

% --------------------------------------------------------------------
% Warehouse Topology (Facts)
% Describes which locations in the warehouse are directly connected
% --------------------------------------------------------------------
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% --------------------------------------------------------------------
% Task 6: Basic Movement Capability Rule
% A robot can move from location X to Y if X and Y are connected.
% --------------------------------------------------------------------
can_move(X, Y) :-
    connected(X, Y).

% --------------------------------------------------------------------
% Task 7: Plan Validation Rules
% Rule to check if a single movement step is valid according to KB
% --------------------------------------------------------------------
valid_move(X, Y) :-
    connected(X, Y).

% Extended rule: Check if a multi-step sequence of moves is valid
valid_plan([]).
valid_plan([move(X, Y)|Rest]) :-
    valid_move(X, Y),
    valid_plan(Rest).

% --------------------------------------------------------------------
% Task 8: Logical Reasoning Chain
% Facts and implication rules for road conditions
% --------------------------------------------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.
