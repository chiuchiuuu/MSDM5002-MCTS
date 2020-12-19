import random
import pickle
import numpy as np


class NBoard(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


# some utility functions
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def relu(X):
    out = np.maximum(X, 0)
    return out


def conv_forward(X, W, b, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.shape
    # theano conv2d flips the filters (rotate 180 degree) first
    # while doing the calculation
    W = W[:, :, ::-1, ::-1]
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_indices(X, h_filter, w_filter,
                           padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = (np.dot(W_col, X_col).T + b).T
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)
    return out


def fc_forward(X, W, b):
    out = np.dot(X, W) + b
    return out


def get_im2col_indices(x_shape, field_height,
                       field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height,
                                 field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


class PolicyValueNetNumpy():
    """policy-value network in numpy """

    def __init__(self, board_width, board_height, net_params):
        self.board_width = board_width
        self.board_height = board_height
        self.params = net_params

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()

        X = current_state.reshape(-1, 4, self.board_width, self.board_height)
        # first 3 conv layers with ReLu nonlinearity
        for i in [0, 2, 4]:
            X = relu(conv_forward(X, self.params[i], self.params[i + 1]))
        # policy head
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.flatten(), self.params[8], self.params[9])
        act_probs = softmax(X_p)
        # value head
        X_v = relu(conv_forward(X, self.params[10],
                                self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value


def load_model(model_path: str = '8_8_5.model'):
    policy_param = pickle.load(open(model_path, 'rb', ), encoding='bytes')
    width, height = 8, 8
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
    return best_policy.policy_value_fn


nn_predict = load_model('zyh_8_8_5.model')

def predict_probs(state):
    nboard = NBoard()
    nboard.init_board(0)
    nboard.current_player = 1 if state.current_player_id == 0 else 2
    for i in range(8):
        for j in range(8):
            c1 = state.board[0][i][j]
            c2 = state.board[1][i][j]
            x, y = 7 - i, j
            move = nboard.location_to_move([x, y])
            if c1 or c2:
                nboard.availables.remove(move)
                if c1 == 0:
                    nboard.states[move] = 2
                else:
                    nboard.states[move] = 1

    if state.last_action:
        nboard.last_move = nboard.location_to_move([7 - state.last_action[0], state.last_action[1]])

    moves, probs = [], []
    result = nn_predict(nboard)
    reward = result[1]
    for move, prob in result[0]:
        loc = nboard.move_to_location(move)
        moves.append((7-loc[0], loc[1]))
        probs.append(prob)

    indexs = list(reversed(np.argsort(probs)))
    moves, probs = np.array(moves)[indexs], np.array(probs)[indexs]
    moves, probs = [(a[0],a[1]) for a in moves], list(probs)
    moves_probs = list(zip(moves,probs))
    return moves_probs, reward