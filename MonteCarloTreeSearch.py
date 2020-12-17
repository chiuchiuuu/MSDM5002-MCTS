from MonteCarloTreeNode import MonteCarloTreeNode
import time
from GomokuGameState import GomokuGameState
import copy
import random
from joblib import Parallel, delayed
import multiprocessing

class MonteCarloTreeSearch:
    """

    """
    def __init__(self, n_iter=1000, parallel=False):
        """
        initialize a Monte Carlo Tree Search Algorithm

        Parameters:
        -----------
        node: MonteCarloTreeNode
            the root node of the MCTS
        """

        self.n_iter = n_iter
        self.root = MonteCarloTreeNode(None)
        self.parallel = parallel

    def update_with_action(self, action):
        """
        move to the next node

        Parameters:
        --------
        action: (int, int)
            last action
        """
        if action in self.root.child:
            self.root = self.root.child[action]
        else:
            self.root = MonteCarloTreeNode(None)

    def _single_run(self, state):
        state_copy = copy.deepcopy(state)

        # get the node to run the simulation
        node = self.select_node(state_copy)
        reward = self.simulate(state_copy)
        node.backpropagate(reward) 

    def run(self, state: GomokuGameState):
        """
        run MCTS algorithm

        Parameters:
        --------
        """
        for _ in range(10):
            self._single_run(state)

        # run simulations
        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(self._single_run)(state) for _ in range(self.n_iter-10))
        else:
            for _ in range(self.n_iter-10):
                self._single_run(state)
        

    def select_node(self, state):
        """
        select a leaf node for the simulation
        """
        current_node = self.root
        #while not current_node.is_terminal_node():
        while not state.is_game_over():
            if current_node.is_leaf():
                action, node = current_node.expand(state)
                state.take_action(action)
                return node
            else:
                action, node = current_node.best_child(c=1)
                state.take_action(action)
                current_node = node

        return current_node

    def simulate(self, state):
        """
        run single simulation (from node to terminal)
        """
        player_id = 1 - state.current_player_id
        while not state.is_game_over():
            actions = state.get_legal_action()
            
            # random 
            action = random.choice(actions)
            state.take_action(action)

        if not state.winner:
            return 0
        else:
            return 1 if state.winner == player_id else -1

    def backpropagate(self, node, winner):
        """
        back propagate the simulation result up to the root node
        """

    def best_action(self):
        """
        Get the best action

        best action is the action of most visited child node
        """
        return max(self.root.child.items(), key=lambda child: child[1].n_visit)[0]
        



if __name__ == "__main__":
    pass


