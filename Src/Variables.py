from typing import TypeVar, Dict, List, Tuple

State = TypeVar('State')
States = List[State]
Transitions = Dict[State,Tuple[State,(int or float)]]