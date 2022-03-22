from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):

    def __init__(self, grid, obey_prob, gamma=0.9):
        self.env = grid
        self.obey_prob = obey_prob
        self.gamma = gamma

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def calculate_policy(self):
        pass

    @abstractmethod
    def calculate_values(self):
        pass
    
    @abstractmethod
    def print_solution(self):
        pass
