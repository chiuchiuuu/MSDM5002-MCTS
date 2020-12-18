import numpy as np
import random
import time
from GomokuGameState import GomokuGameState
import copy
from joblib import Parallel, delayed
import multiprocessing

class MonteCarloTreeNode:
    """
    
    """
    def __init__(self, parent):
        """
        initilize for Monte Carlo Tree Node

        Parameters:
        -----------
        state: (n, n) numpy array
        parent: parent tree node
        """

        self.child = dict() # action:childnode
        self.parent = parent

        self.is_visted = False

        self._untried_actions = None


        # node's statistics
        self.n_win = 0
        self.n_lose = 0
        self.n_visit = 0


    def expand(self, state):
        """
        return an unvisited child nodes
        """
        if not self._untried_actions:
            self._untried_actions = state.get_legal_action()

        #action = self._untried_actions.pop()

        action = random.choice(self._untried_actions)
        self._untried_actions.remove(action)

        self.child[action] = MonteCarloTreeNode(self)
        return action, self.child[action]

    def is_leaf(self):
        """
        check if a node is a leaf node

        a node is leaf if it is not fully expanded, or exists untried action
        """
        return (self._untried_actions is None) or len(self._untried_actions)

    # def is_terminal_node(self):
    #     """
    #     check if a node is terminal node
    #     """

    def best_child(self, c=1):
        """
        return the best child of current node

        best action is decided by utc funtion
        """
        return max(self.child.items(), key = lambda child: child[1].uct(c))

    def uniform_random_rollout_policy(self, nodes):
        """
        unifrom random roll out policy for the simulation

        Parameters:
        --------
        nodes: list
            list of optional nodes
        """
        return np.random.choice(nodes)


    def uct(self, c=np.sqrt(2)):
        """
        Compute the upper confidence bound of current node

        Parameters:
        --------
        c: float
            constant for the UCT function
        """
        return self._Q() / self._N() + c * np.sqrt(np.log(self.parent._N()) / self._N())

    def _Q(self):
        """
        reward of a node
        """
        return self.n_win - self.n_lose
        # return self.n_win

    def _N(self):
        """
        total number of visits
        """
        return self.n_visit

    def backpropagate(self, reward):
        self.n_win += (reward==1)
        self.n_lose += (reward==-1)
        self.n_visit +=1 

        if self.parent:
            self.parent.backpropagate(-reward)
        


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
            tmp = MonteCarloTreeNode(self.root)
            self.root.child[action] = tmp
            self.root = tmp

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