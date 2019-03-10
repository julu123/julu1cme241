from Processes.Variables import State, States, Action
from Utils.GBMStockSimulator import GMBStockSimulator
from typing import Callable, Tuple


class FunctionApproximationBase:

    def __init__(self,
                 generator_function: Callable[[State, Action], Tuple[State, float]],
                 terminal_states: Callable[[State], bool],
                 gamma: float = 0.99):
        self.gamma = gamma

    def generate(self):
        pass
