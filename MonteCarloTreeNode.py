import numpy as np
import random

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


    def uct(self, c=1):
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
        


if __name__ == "__main__":
    pass