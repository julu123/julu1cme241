from typing import TypeVar, Dict, List, Tuple

# MP
State = TypeVar('State')
States = List[State]
Transitions = Dict[State, Tuple[State, (int or float)]]

# MRP_A
Transitions_rewards = Dict[Transitions, (float or int)]
R_A = List[float]

# MRP_B
PR = Tuple[(float or int), (float or int)]
Transitions_rewards_B = Dict[State, Dict[State, PR]]
R_B = List[List[(float or int)]]

# MDP_A
Action = TypeVar('Action')
Transitions_Rewards_Action_A = Dict[State, Dict[Action, Dict[Dict[State, (float or int)], (float or int)]]]
Policy = Dict[State, Dict[Action, (float or int)]]

# MDP_B
Transitions_Rewards_Action_B = Dict[State, Dict[Action, Tuple[State, PR]]]

# Tabular_base
SA = Tuple[State, Action]
SR = Tuple[State, (float or int)]

# Function Approximation Base
#Feature = TypeVar('Feature')
#Features = List[Tuple[Feature, (float or int)]]
#SF = Dict[Tuple[State, Features]]
