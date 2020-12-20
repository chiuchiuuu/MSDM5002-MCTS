import pygame
import numpy as np
import random
from GomokuGameState import GomokuGameState
from GomokuGamePlayer import MCTSPlayer, HumanPlayer, MCTSPlayerAlpha

class Gomoku:

    def __init__(self, size, player1, player2, gui=True):
        """
        initilize gomuku game

        Parameters:
        -----------
        size: int
            size of the board
        player1, player2: GomokuGamePlayer
            the two players of the Gomoku game
        gui: bool
            True if you want a graphical user interface
        """
        self.size = size
        self.gui = gui
        self.state = GomokuGameState(self.size, (player1, player2), start_player=0)

        if gui:
            # pygame
            pygame.init()
            self.screen = pygame.display.set_mode((640,640))
            pygame.display.set_caption('Five-in-a-Row')
            self._draw_board()
            pygame.display.update()

    def run(self):
        """
        run the Gomoku game
        """
        while not self.state.is_game_over():
            player = self.state.get_current_player()
            action = player.get_action(self.state)
            self.state.take_action(action)

            if self.gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                self.render()

        ## handling game end
        if self.state.winner is not None:
            message = f"Player {self.state.winner} wins!"
        else:
            message = "Draw!"

        if self.gui:
            font = pygame.font.Font('freesansbold.ttf', 32)
            text = font.render(message, True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (320,320)
            while True:
                self.screen.blit(text, text_rect)
                pygame.display.update()
                for event in pygame.event.get():
                     if event.type == pygame.MOUSEBUTTONDOWN:
                         pygame.quit()
                         return
        else:
            print(message)


    def collect_play_data(self):
        """
        run and collect data
        """
        boards = []
        players = []
        probs = []
        z = []
        while not self.state.is_game_over():
            player = self.state.get_current_player()
            action, action_prob = player.get_action(self.state, return_prob=True)

            ## store data
            boards.append(self.state.get_board_under_current_player())
            players.append(self.state.current_player_id)

            action_id = [ac[0]*self.size+ac[1] for ac in action_prob.keys()]
            prob = np.zeros(self.size*self.size)
            prob[action_id] = list(action_prob.values())
            probs.append(prob)

            self.state.take_action(action)

            if self.gui:
                self.render()

        z = np.zeros(len(players))
        if self.state.winner is not None:
            z[np.array(players) == self.state.winner] = 1
            z[np.array(players) != self.state.winner] = -1
        
        return zip(boards, probs, z)

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
        d=int(560/(self.size-1))
        black_color = [0, 0, 0]
        #board_color = [ 241, 196, 15]
        board_color = [ 238,216,174]
        self.screen.fill(board_color)
        pygame.draw.circle(self.screen, black_color,(120,120), 6)
        pygame.draw.circle(self.screen, black_color,(520,520), 6)
        pygame.draw.circle(self.screen, black_color,(520,120), 6)
        pygame.draw.circle(self.screen, black_color,(120,520), 6)
        for h in range(0, self.size):
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

        board = self.state.get_board()
    
        d=int(560/(self.size-1))
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i,j]==1:
                    pos = [40+d * j, 40+d* i ]
                    pygame.draw.circle(self.screen, black_color, pos, 18,0)
                elif board[i,j]==-1:
                    pos = [40+d* j , 40+d * i]
                    pygame.draw.circle(self.screen, white_color, pos, 18,0)

if __name__ == '__main__':
    # gomoku = Gomoku(8, HumanPlayer(), MCTSPlayer(max_time=10), gui=True)
    # gomoku.run()

    # gomoku = Gomoku(size=6, self_run=True, gui=True, max_time=5)
    # results = gomoku.run_self_play()
    # results = list(results)
    # for board, prob, z in results:
    #     print(z)
    from PolicyValueNet import PolicyValueNetNumpy
    import pickle

    net_param = pickle.load(open('zyh_8_8_5.model', 'rb', ), encoding='bytes')
    policy_value_fn = PolicyValueNetNumpy(8, 8, net_param).policy_value_fn

    gomoku = Gomoku(8, MCTSPlayer(max_time=10), MCTSPlayerAlpha(policy_value_fn,max_time=10))
    gomoku.run()