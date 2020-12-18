import numpy as np
import copy

class GomokuGameState:
    """
    game state for gomoku
    """
    def __init__(self, size, players, start_player=0):
        """
        initialize a Gomoku game state

        Parameters:
        --------
        size: int
            size of the board
        player: int, (0, 1)
            player of  current state
        """
        self.size = size
        self.board = np.zeros((4, self.size, self.size), dtype=int)
        self.players = players
        self.current_player_id = start_player
        self.last_action = None
        self.winner = None

        self._legal_actions = list(range(size*size)) # store row-wise as integers

    def __deepcopy__(self, memodict={}):
        state_copy = GomokuGameState(self.size, self.players)

        state_copy.current_player_id = self.current_player_id
        state_copy.board = np.copy(self.board)
        state_copy.last_action = self.last_action
        state_copy.winner = self.winner
        state_copy._legal_actions = self._legal_actions.copy()
        return state_copy


    def take_action(self, action):
        """
        take action for current state

        Parameters:
        action: (int, int)
            the location to take place
        """
        self.board[self.current_player_id, action[0], action[1]] = 1

        if self.last_action:
            self.board[self.current_player_id+2] = np.zeros((self.size, self.size))
            self.board[self.current_player_id+2, self.last_action[0], self.last_action[1]] = 1

        # change player
        self.current_player_id = 1 - self.current_player_id
        # update last action
        self.last_action = action
        # remove legal actions
        self._legal_actions.remove(action[0]*self.size+action[1])

    def get_current_player(self):
        """
        return current player
        """
        return self.players[self.current_player_id]

    def is_game_over(self):
        """
        check if the game is over
        """
        if len(self._legal_actions) == 0:
            return True
        if self.last_action is None:
            return False

        last_player_id = 1 - self.current_player_id
        board = self.board[last_player_id]

        i, j = self.last_action
        # check column
        start, end = max(i-4, 0), min(i, self.size-1-4)
        for k in range(start, end+1):
            if board[k:k+5,j].all():
                self.winner = last_player_id
                return True

        # check rows
        start, end = max(j-4,0), min(j, self.size-1-4)
        for k in range(start, end+1):
            if board[i, k:k+5].all():
                self.winner = last_player_id
                return True

        # check diagonal
        if '11111' in ''.join(np.diagonal(board, j-i).astype(str)):
            self.winner = last_player_id
            return True

        # check reverse diagonal
        if '11111' in ''.join(np.fliplr(board).diagonal(self.size-j-i-1).astype(str)):
            self.winner = last_player_id
            return True

        return False

    def get_legal_action(self):
        """
        get all legal actions under current state

        Returns:
        --------
        actions: list of (int, int)
            list of legal actions
        """
        # convert integer to (row, col)
        return [(t//self.size, t%self.size) for t in self._legal_actions]

    def is_legal_action(self, action):
        """
        check if an action is legal

        Parameters:
        -------
        action: (int, int)
        """
        return action in self.get_legal_action()

    def get_board(self):
        """
        return a 2d array form of board, 1 for player 0 and -1 for player 1
        """
        board = np.zeros((self.size, self.size))
        board[self.board[0]==1] = 1
        board[self.board[1]==1] = -1
        return board


    def get_board_under_current_player(self):
        """
        get board data under current player
        """
        a, b = self.current_player_id, 1 - self.current_player_id
        board_cp = self.board[[a,b,a+2,b+2],:,:].copy()
        return board_cp