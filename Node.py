import numpy as np

class Node():

    def __init__(self, state, parent,action,player,is_root,board):
        self.visited_count = 0
        self.win_count = 0
        self.state = state
        self.action = action
        self.parent = parent
        self.player = player   # 1 代表黑棋 2 代表白棋
        self.child = []
        self.simulation_count = 0
        self.is_root = is_root
        self.board = board

    def update_node(self,visit_count_update,win_count_update):

        self.visited_count += visit_count_update
        self.win_count += win_count_update
