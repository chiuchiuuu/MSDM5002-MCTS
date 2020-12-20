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
        self.board = np.zeros((2, self.size, self.size), dtype=int)
        self.players = players
        self.current_player_id = start_player
        self.last_action = None
        self.winner = None
        self.last_last_action = None

        self._legal_actions = list(range(size*size)) # store row-wise as integers

    def __deepcopy__(self, memodict={}):
        state_copy = GomokuGameState(self.size, self.players)

        state_copy.current_player_id = self.current_player_id
        state_copy.board = np.copy(self.board)
        state_copy.last_action = self.last_action
        state_copy.last_last_action = self.last_last_action
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

        #if self.last_last_action:
        #    self.board[self.current_player_id+2] = 0
        #    self.board[self.current_player_id+2, self.last_last_action[0], self.last_last_action[1]] = 1

        # change player
        self.current_player_id = 1 - self.current_player_id
        # update last action
        self.last_last_action = self.last_action
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
            # if board[k:k+5, j].all():
            if board[k,j] and board[k+1,j] and board[k+2,j] and board[k+3,j] and board[k+4,j]:
                self.winner = last_player_id
                return True  

        # check rows
        start, end = max(j-4,0), min(j, self.size-1-4)
        for k in range(start, end+1):
            # if board[i, k:k+5].all():
            if board[i,k] and board[i,k+1] and board[i,k+2] and board[i,k+3] and board[i,k+4]:
                self.winner = last_player_id
                return True

        # check diagonal
        # if '11111' in ''.join(np.diagonal(board, j-i).astype(str)):
        #     self.winner = last_player_id
        #     return True
        if i <= j:
            offset = j-i
            start, end = max(i-4, 0), min(i, self.size-1-4-offset)
            for k in range(start, end+1):
                # if board[range(k,k+5), range(k+(j-i),k+(j-i)+5)].all():
                if board[k,k+offset] and board[k+1,k+1+offset] and board[k+2,k+2+offset] and \
                    board[k+3,k+3+offset] and  board[k+4, k+4+offset]:
                    self.winner = last_player_id
                    return True
        if i > j:
            offset = (i-j)
            start, end = max(j-4, 0), min(j, self.size-1-4-offset)
            for k in range(start, end+1):
                # if board[range(k+(i-j),k+(i-j)+5), range(k,k+5)].all():
                if board[k+offset,k] and board[k+1+offset,k+1] and board[k+2+offset,k+2] and \
                    board[k+3+offset,k+3] and board[k+4+offset,k+4]:
                    self.winner = last_player_id
                    return True

        # check reverse diagonal
        # if '11111' in ''.join(np.fliplr(board).diagonal(self.size-j-i-1).astype(str)):
        #     self.winner = last_player_id
        #     return True

        if i+j+1>=5 and i + j + 1 <= self.size: # i+j+1 is the size of the reverse diagonal
            start, end = max(i-4,0), min(i, i+j-4)
            for k in range(start, end+1):
                if board[k,i+j-k] and board[k+1,i+j-k-1] and board[k+2,i+j-k-2] and board[k+3,i+j-k-3] \
                    and board[k+4,i+j-k-4]:
                    self.winner = last_player_id
                    return True
        elif (i + j + 1 > self.size) and (2*self.size-i-j-1 >= 5): # size: self.size-(i+j+1-self.size)
            start, end = max(i-4, i+j+1-self.size), min(i+4, max(0,self.size-5))
            for k in range(start, end+1):
                if board[k,i+j-k] and board[k+1,i+j-k-1] and board[k+2,i+j-k-2] and board[k+3,i+j-k-3] \
                    and board[k+4,i+j-k-4]:
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

    def view_board(self):
        board = np.zeros((self.size, self.size),dtype=int).astype(str)
        board[:,:] = '_'
        board[self.board[0]==1] = 'x'
        board[self.board[1]==1] = 'o'
        return board


    def get_board_under_current_player(self):
        """
        get board data under current player
        """
        # a, b = self.current_player_id, 1 - self.current_player_id
        # board_cp = self.board[[a,b,a+2,b+2,4],:,:].copy()
        # board_cp[4] = (self.current_player_id == 0)
        # return board_cp

        a, b = self.current_player_id, 1 - self.current_player_id
        board_cp = np.zeros((4,self.size,self.size))
        board_cp[:2,:,:] = self.board[[a,b],:,:].copy()
        board_cp[2, self.last_action[0], self.last_action[1]] = 1
        board_cp[2] = (self.current_player_id == 0)
        return board_cp
