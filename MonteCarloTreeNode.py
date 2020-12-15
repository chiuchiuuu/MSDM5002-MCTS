import numpy as np
import random

class MonteCarloTreeNode:
    """
    
    """
    def __init__(self, state, player == -1, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.N = 0
        self.value = 0

    def untried_action(self) #? uncomplete
        M = self.state.shape(0)
        choice = []
        for row in range(M):
            for col in range(M):
                if self.state[row,col] == 0:
                    round = []
                    for r in range(row-1,row+2):
                        for c in range(col-1,col+2):
                            round.append((r,c))
                    around = self.state[tuple(round)]
                    if 1 in around or -1 in around:
                        choice.append((row,col))
        return tuple(choice)
    
    def expend(self):
        target = self._untried_action()[0]
        state = self.state
        state[tuple(target)] = self.player
        n = MonteCarloTreeNode(state, 0-self.player, parent=self)

    def is_terminal_node(self):
        if self.state.gameover:
            return True
        return False

    def rollout(self):
        state = self.state
        player = self.player
        row, col = np.where(state==0)
        idx = np.random.randint(len(row))
        while not state.gameover:
            step += 1
            row, col = np.where(state==0)
            idx = np.random.randint(len(row))
            state[row[idx], col[idx]] = player
            player = -player
        remain = np.where(state==0)+1
        if state.gameover == self.player:
            return 5.0*len(remain)
        else:
            return 3.0/len(remain)

    def backpropagate(self, reward):
        value = reward.rollout()
        t = reward
        while t.parent:
            t.value += value
            t.N += 1
            t = t.parent
        self.value += value
        self.N += 1

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        UCT = [(c.value / c.N) + c_param * np.sqrt((2 * np.log(self.N) / c.N)) for c in self.children]
        return self.children[np.argmax(UCT)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]