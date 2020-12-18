import time
import numpy as np
import pygame
from MonteCarloTreeNode import MonteCarloTreeNode
from MonteCarloTreeSearch import MonteCarloTreeSearch
from setting import *

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

    def _update_by_pc(self):
        """
        This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game 
        with a human
        input:
            2D matrix representing the state of the game.
        output:
            2D matrix representing the updated state of the game.
        """
        
        row, col = np.where(self.mat==0)
        idx = np.random.randint(len(row))
        self.mat[row[idx], col[idx]] = -1
        pass

    # def update_by_man(self):
    #     """
    #     This function detects the mouse click on the game window. Update the state matrix of the game. 
    #     input: 
    #         event:pygame event, which are either quit or mouse click)
    #         mat: 2D matrix represents the state of the game
    #     output:
    #         mat: updated matrix
    #     """
    #     done=False
    #     if event.type==pygame.QUIT:
    #         done=True
    #     if event.type==pygame.MOUSEBUTTONDOWN:
    #         (x,y)=event.pos
    #         row = round((y - 40) / 40)     
    #         col = round((x - 40) / 40)
    #         mat[row][col]=1
    #     return mat, done

    def run(self):
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
                    self.render()
                     # check for win or tie
                    # print message if game finished
                    # otherwise contibue
                    
                    
                    #get the next move from computer/MCTS
                    time.sleep(0.3)
                    self._update_by_pc()
                    self.render()
                    # check for win or tie
                    # print message if game finished
                    # otherwise contibue

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

if __name__ == '__main__':
    gomoku = Gomoku(8)
    gomoku.run()