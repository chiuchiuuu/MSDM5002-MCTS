import numpy as np
import random
import time
from GomokuGameState import GomokuGameState
import copy
from joblib import Parallel, delayed
import multiprocessing
from typing import Dict

class MonteCarloTreeNode:
    """
    
    """
    def __init__(self, parent, prior_prob=1):
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

        self.prior_prob = prior_prob


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

    def is_leaf_alpha(self):
        return len(self.child)== 0


    def best_child(self):
        """
        return the best child of current node

        best action is decided by utc funtion
        """
        #tmp = [child.uct() for child in self.child.values()]
        #print(tmp)
        return max(self.child.items(), key = lambda child: child[1].uct())

    def best_child_alpha(self):
        """
        return the best child by uct_alpha
        """
        return max(self.child.items(), key = lambda child: child[1].uct_alpha())


    def uct(self, c_puct=np.sqrt(2)):
        """
        Compute the upper confidence bound of current node

        Parameters:
        --------
        c: float
            constant for the UCT function
        """
        # return self._Q() / self._N() + c_puct * np.sqrt(np.log(self.parent._N()) / self._N())
        return (self.n_win-self.n_lose)/self.n_visit + c_puct * np.sqrt(np.log(self.parent.n_visit) / self.n_visit)

    def uct_alpha(self, c_puct=5):
        """
        compute uct function of alpha zero
        """
        _Q = (self.n_win-self.n_lose)/self.n_visit if self.n_visit > 0 else 0
        return _Q + \
            c_puct * self.prior_prob * np.sqrt(self.parent.n_visit/(1 + self.n_visit))

    def backpropagate(self, reward):
        self.n_win += (reward==1)
        self.n_lose += (reward==-1)
        self.n_visit += 1

        if self.parent:
            self.parent.backpropagate(-reward)

    def full_expand(self, action_probs):
        """
        fully expand current node, assign all child nodes with a prior probability
        """
        for action, prob in action_probs:
            if action not in self.child:
                self.child[action] = MonteCarloTreeNode(self, prior_prob=prob)

class MonteCarloTreeSearch:
    """
    
    """
    def __init__(self, n_iter=20000, parallel=False, max_time=None, type="pure", policy_network=None):
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
        self.max_time = max_time

        self.type = type
        self.policy_network = policy_network


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

    def run(self, state: GomokuGameState):
        """
        run MCTS algorithm

        Parameters:
        --------
        """
        # run simulations
        #if self.parallel:
        #    num_cores = multiprocessing.cpu_count()
        #    Parallel(n_jobs=num_cores)(delayed(self._single_run)(state) for _ in range(self.n_iter-10))
        #else:

        if self.type == 'pure':
            start_time = time.time()
            for _ in range(self.n_iter):
                if self.max_time and (time.time() - start_time > self.max_time):
                    print(f"number of iteration: {_}")
                    break
                state_copy = copy.deepcopy(state)
                # get the node to run the simulation
                node = self.select_node(state_copy)
                reward = self.simulate(state_copy)
                node.backpropagate(reward)
            return
        elif self.type == 'alphazero':
            start_time = time.time()
            for _ in range(self.n_iter):
                state_copy = copy.deepcopy(state)
                self.playout(state_copy)
                
    def playout(self, state):
        """
        run a single playout for alphazero mcts
        """
        node = self.root            
        while not node.is_leaf_alpha():
            action, node = node.best_child_alpha()
            state.take_action(action)

        if not state.is_game_over():
            action_probs, reward = self.policy_network(state)
            node.full_expand(action_probs)
        else:
            if state.winner is None:
                reward = 0
            else:
                reward = 1 if state.winner == state.current_player_id else -1

        node.backpropagate(reward)



    def select_node(self, state):
        """
        select a leaf node for the simulation
        """
        current_node = self.root
        while not current_node.is_leaf():
            action, current_node = current_node.best_child()
            state.take_action(action)

        if state.is_game_over():
            return current_node
        else:
            action, node = current_node.expand(state)
            state.take_action(action)
            return node

        # while not state.is_game_over():
        #     if current_node.is_leaf():
        #         action, node = current_node.expand(state)
        #         state.take_action(action)
        #         return node
        #     else:
        #         action, node = current_node.best_child()
        #         state.take_action(action)
        #         current_node = node

        # return current_node

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

        # if not state.winner
        if state.winner is None:
            #print("simulation: draw")
            return 0
        else:
            #print(f"simulation: {state.winner} wins")
            return 1 if state.winner == player_id else -1

    def best_action(self):
        """
        Get the best action

        best action is the action of most visited child node
        """
        return max(self.root.child.items(), key=lambda child: child[1].n_visit)[0]


    def get_action_probability(self, temp=1):
        """
        get the probability for each action of self.root

        Parameters:
        --------
        temp: float
            temperature parameters

        Returns:
        --------
        action_probs: dict(action, probability)
        """

        action_visit = {child[0]:child[1].n_visit for child in self.root.child.items()}

        sum_visit = sum([visit**(1/temp) for visit in action_visit.values()])

        action_prob = action_visit.copy()
        for action in action_visit:
            action_prob[action] = (action_visit[action]**(1/temp) / sum_visit)

        return action_prob


if __name__ == "__main__":
    pass