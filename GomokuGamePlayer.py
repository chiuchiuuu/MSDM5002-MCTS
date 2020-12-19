from abc import ABC, abstractmethod
import pygame
import numpy as np
import random
import sys
from MCTS import MonteCarloTreeSearch


class GomokuGamePlayer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, state):
        """
        get the player's action for current state
        """
        pass

class MCTSPlayer(GomokuGamePlayer):
    def __init__(self, n_iter=20000, parallel=False, max_time=None):
        super().__init__()
        self.mcts = MonteCarloTreeSearch(n_iter=n_iter, parallel=parallel, max_time=max_time)
        self.tree = self.mcts.root
        
    def get_action(self, state, return_prob=False):
        """
        Return an action for the current state
        """
        # random action
        # actions = state.get_legal_action()
        # return tuple(random.choice(actions))

        self.mcts.update_with_action(state.last_action)
        self.mcts.run(state)
        action = self.mcts.best_action()
        action_prob = self.mcts.get_action_probability()

        self.mcts.update_with_action(action)

        if return_prob:
            return action, action_prob
        else:
            return action

class HumanPlayer(GomokuGamePlayer):
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        while True:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type==pygame.MOUSEBUTTONDOWN:
                    (x,y)=event.pos
                    d=int(560/(state.size-1))
                    row = round((y - 40) / d)
                    col = round((x - 40) / d)

                    action = (row, col)

                    if state.is_legal_action(action):
                        return action


class MCTSPlayerAlpha(GomokuGamePlayer):
    """
    A player with alphazero version of MCTS
    """
    def __init__(self, policy_network, n_iter):
        self.mcts = MonteCarloTreeSearch(n_iter=n_iter, type="alpha", policy_network=policy_network)

    def get_action(self, state, return_prob=False):
        """
        Return an action for the current state
        """
        # random action
        # actions = state.get_legal_action()
        # return tuple(random.choice(actions))

        self.mcts.update_with_action(state.last_action)
        self.mcts.run(state)
        action_prob = self.mcts.get_action_probability()
        action = max(action_prob.items(), key = lambda x: x[1])[0]

        self.mcts.update_with_action(action)

        if return_prob:
            return action, action_prob
        else:
            return action