import time
import numpy as np
import pygame
from setting import *
import copy
from MentecarloTreeSearch import MentecarloTreeSearch
from Node import Node
# from MCTS import MentecarloTreeSearch

class Gomoku:

    def __init__(self, M):
        """
        initilize gomuku game
        Parameters:
        -----------
        M: int
            size of the board
        """

        # game state
        self.M = M
        self.mat = np.zeros((M,M))
        self.done = False
        self.MCTS = MentecarloTreeSearch(False, self.M, self.mat)

        # pygame
        pygame.init()
        self.screen = pygame.display.set_mode((640,640))
        pygame.display.set_caption('Five-in-a-Row')
        self._draw_board()
        pygame.display.update()

    def _check_for_done(self, i, j):
        """
        please write your own code testing if the game is over. Return a boolean variable done. If one of the players wins
        or the tie happens, return True. Otherwise return False. Print a message about the result of the game.
        input: 
            2D matrix representing the state of the game
        output:
            none
        """
        
        # return done
        pass

    def _update_by_pc(self, node):
        """
        This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game 
        with a human
        input:
            2D matrix representing the state of the game.
        output:
            2D matrix representing the updated state of the game.
        """

        t_mat = copy.deepcopy(self.mat)
        action, pl, next_child = self.MCTS.run(node)
        self.mat[action[0], action[1]] = pl

        return next_child


    def run(self):
        run_count = 0
        while not self.done:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    self.done=True
                if event.type==pygame.MOUSEBUTTONDOWN:
                    (x,y)=event.pos
                    d=int(560/(self.M-1))
                    row = round((y - 40) / d)     
                    col = round((x - 40) / d)
                    self.mat[row][col]=1
                    run_count += 1
                    self.render()
                    win_pl, win = self.win_condition(self.mat)
                    if win:
                        self.done = True
                        break

                    if run_count == 1:
                        tmp_node = Node(self.mat, False, False, 1, True, 1)
                        self.MCTS.root = tmp_node
                        self.MCTS.mat = self.mat
                    else:
                        new_node = Node(self.mat, tmp_node, [row, col], tmp_node.player * (-1), False, 1)
                        for child_node in tmp_node.child:
                            if ((row == child_node.action['position'][0]) and (col == child_node.action['position'][1])):
                                new_node = child_node
                                break
                        tmp_node = new_node

                    # check for win or tie
                    # print message if game finished
                    # otherwise contibue
                    
                    
                    #get the next move from computer/MCTS
                    time.sleep(0.3)
                    new_node = self._update_by_pc(tmp_node)
                    tmp_node = new_node
                    self.render()
                    run_count += 1
                    win_pl, win = self.win_condition(self.mat)
                    if win:
                        self.done = True
                    # check for win or tie
                    # print message if game finished
                    # otherwise contibue
        print(f'{win_pl} wins!')
        pygame.quit()

    def render(self):
        """
        Draw the updated game with lines and stones using function draw_board and draw_stone
        input:
            screen: game window, onto which the stones are drawn
            mat: 2D matrix representing the game state
        output:
            none        
        """
        self._draw_board()
        self._draw_stone()
        pygame.display.update()


    def _draw_board(self):    
        """
        This function draws the board with lines.
        input: game windows
        output: none
        """
        d=int(560/(self.M-1))
        black_color = [0, 0, 0]
        board_color = [ 241, 196, 15]
        self.screen.fill(board_color)
        for h in range(0, self.M):
            pygame.draw.line(self.screen, black_color,[40, h * d+40], [600, 40+h * d], 1)
            pygame.draw.line(self.screen, black_color, [40+d*h, 40], [40+d*h, 600], 1)

    def _draw_stone(self):
        """
        This functions draws the stones according to the mat. It draws a black circle for matrix element 1(human),
        it draws a white circle for matrix element -1 (computer)
        input:
            screen: game window, onto which the stones are drawn
            mat: 2D matrix representing the game state
        output:
            none
        """
        black_color = [0, 0, 0]
        white_color = [255, 255, 255]
        M=len(self.mat)
        d=int(560/(M-1))
        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):
                if self.mat[i,j]==1:
                    pos = [40+d * j, 40+d* i ]
                    pygame.draw.circle(self.screen, black_color, pos, 18,0)
                elif self.mat[i,j]==-1:
                    pos = [40+d* j , 40+d * i]
                    pygame.draw.circle(self.screen, white_color, pos, 18,0)

    @staticmethod
    def win_condition(t_mat):

        width, height = t_mat.shape[0], t_mat.shape[1]

        for i in range(0, width):
            for j in range(0, height):
                player = int(t_mat[i][j])

                if player != 0:
                    if i <= width - 1 - 4:
                        chess_set = set(t_mat[i + k][j] for k in range(0, 5))
                        # 横右向
                        if len(chess_set) == 1:
                            return player, True
                    if j <= height - 1 - 4:
                        # 纵下向
                        chess_set = set(t_mat[i][j + k] for k in range(0, 5))
                        if len(chess_set) == 1:
                            return player, True
                    if (i <= width - 4 - 1) and (j <= height - 4 - 1):
                        # 右下
                        chess_set = set(t_mat[i + k][j + k] for k in range(0, 5))
                        if len(chess_set) == 1:
                            return player, True

                    if (i <= width - 4 - 1) and (j >= 5):
                        # 右上
                        chess_set = set(t_mat[i + k][j - k] for k in range(0, 5))
                        if len(chess_set) == 1:
                            return player, True

                    if (i >= 5) and (j >= 5):
                        # 左上
                        chess_set = set(t_mat[i - k][j - k] for k in range(0, 5))
                        if len(chess_set) == 1:
                            return player, True

                    if (i >= 5) and (j <= height - 4 - 1):
                        # 左下
                        chess_set = set(t_mat[i - k][j + k] for k in range(0, 5))
                        if len(chess_set) == 1:
                            return player, True

        return player, False
if __name__ == '__main__':
    gomoku = Gomoku(8)
    gomoku.run()

