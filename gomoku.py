import numpy as np
import pygame
from gui import *
from MCTS import *
from setting import *


def check_for_done(mat):
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

def update_by_pc(mat):
    """
    This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game 
    with a human
    input:
        2D matrix representing the state of the game.
    output:
        2D matrix representing the updated state of the game.
    """
    # return mat
    pass

def main():
    
    global M
    #M=8
    
    pygame.init()
    screen=pygame.display.set_mode((640,640))
    pygame.display.set_caption('Five-in-a-Row')
    done=False
    mat=np.zeros((M,M))
    d=int(560/(M-1))
    draw_board(screen)
    pygame.display.update()

    
    while not done:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                done=True
            if event.type==pygame.MOUSEBUTTONDOWN:
                (x,y)=event.pos
                row = round((y - 40) / d)     
                col = round((x - 40) / d)
                mat[row][col]=1
                render(screen, mat)
                # check for win or tie
                # print message if game finished
                # otherwise contibue
                
                
                #get the next move from computer/MCTS
                # check for win or tie
                # print message if game finished
                # otherwise contibue
    
    pygame.quit()
    
if __name__ == '__main__':
    main()