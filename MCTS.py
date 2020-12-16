import numpy as np
from Node import Node
import random
import copy

class MentecarloTreeSearch():

    def __init__(self, root, M, mat):
        self.root = root  # root node
        self.M = M  # length of the chess bored
        self.mat = mat  # board value  initial_value = np.zeros((M,M))

    def expand_node(self, node,expand_node_num):

        mat = copy.deepcopy(node.state)
        child_node_player = node.player * (-1)
        row, col = np.where(mat == 0)
        try:
            child_nd = random.sample(list(range(len(row))), expand_node_num)
        except ValueError:
            print(len(row))
            print(mat)
        for child in child_nd:
            mat_tmp = copy.deepcopy(node.state)
            mat_tmp[row[child]][col[child]] = child_node_player
            action = {'position': [row[child], col[child]],
                      'player': child_node_player
                      }
            node.child.append(Node(mat_tmp, node, action, child_node_player, False))

    def expand_node_2(self, node, expand_node_num):
        t_mat = copy.deepcopy(node.state)
        child_node_player = node.player * (-1)
        row, col = np.where(t_mat == 0)
        child_nd = random.sample(list(range(len(row))), min(expand_node_num-len(node.child), len(row)))
        for child in child_nd:
            mat_tmp = copy.deepcopy(node.state)
            mat_tmp[row[child]][col[child]] = child_node_player
            action = {'position': [row[child], col[child]],
                        'player': child_node_player
                        }
            node.child.append(Node(mat_tmp, node, action, child_node_player, False))

    def process_2(self, node):
        if (node.is_root) and (len(node.child) == 0):
            self.expand_node_2(node,40)
        else:
            self.expand_node_2(node,20)

    def process(self, node):
        if (node.is_root) and (len(node.child) == 0):
            self.expand_node(node,40)
        else:
            if (node.visited_count != 0) and (len(node.child) == 0):
                self.expand_node(node,3)

    def choose_node_2(self, node):

        if len(node.child) != 0:
            best_node = node.child[0]
            uct_value = self.UCT_cal(best_node, node.simulation_count)
            for child_node in node.child:
                # print(f' 当前node的visit次数{child_node.visited_count} 当前node的win次数{child_node.win_count} ')
                tmp_uct = self.UCT_cal(child_node, node.simulation_count)
                # print(f'当前的 UCT value {tmp_uct}')
                if tmp_uct > uct_value:
                    best_node = child_node

        return best_node

    def choose_node(self, node):
        '''
        choose a node to do simulation
        '''

        self.process(node)

        # uct_value放在此处 存疑 为何不行
        if len(node.child) != 0:
            best_node = node.child[0]
            uct_value = self.UCT_cal(best_node, node.simulation_count)
            for child_node in node.child:
                # print(f' 当前node的visit次数{child_node.visited_count} 当前node的win次数{child_node.win_count} ')
                tmp_uct = self.UCT_cal(child_node, node.simulation_count)
                # print(f'当前的 UCT value {tmp_uct}')
                if tmp_uct > uct_value:
                    best_node = child_node


            return self.choose_node(best_node)
        else:
            return node

    def run_2(self, node):
        iter_num = 100
        self.process_2(node)
        for i in range(iter_num):
            t_best_child = self.choose_node_2(node)
            self.simulation_2(t_best_child)
            node.simulation_count += 1
        action_tmp, next_child = self.take_action(node)
        action_position = action_tmp['position']
        player = action_tmp['player']

        return action_position, player, next_child

    def run(self, node):
        iter_num = 100 # 每步棋 迭代次数
        for i in range(iter_num):
            # print(f'第{i + 1}次模拟，当前的simulation count为{node.simulation_count}')
            simulation_node = self.choose_node_2(node)
            self.simulation_2(simulation_node)
            node.simulation_count += 1
            # print('============= ============= ============= =============')
        action_tmp = self.take_action(node)
        action_position = action_tmp['position']
        player = action_tmp['player']

        return action_position, player

    def simulation_2(self, node):
        tmp_mat = copy.deepcopy(node.state)
        win = False

        tmp_player = node.player * (-1)
        while not win:
            tmp_mat = copy.deepcopy(node.state)
            tmp_player = node.player * (-1)

            row, col = np.where(tmp_mat == 0)
            idx = np.random.randint(len(row))
            tmp_mat[row[idx], col[idx]] = tmp_player
            pl, win = self.win_condition(tmp_mat)

            if win:
                self.back_propagation(node, pl)
            elif len(row) == 1:
                self.back_propagation(node, pl)
                break
            else:
                action = {'position': [row[idx], col[idx]],
                          'player': tmp_player
                          }
                tmp_node = Node(tmp_mat, node, action, tmp_player, False)
                node.child.append(tmp_node)
                node = tmp_node


    def simulation(self, node):
        tmp_mat = copy.deepcopy(node.state)
        win = False

        tmp_player = node.player * (-1)
        while not win:
            row, col = np.where(tmp_mat == 0)
            idx = np.random.randint(len(row))
            tmp_mat[row[idx], col[idx]] = tmp_player
            pl, win = self.win_condition(tmp_mat)

            if len(row) == 1:
                break
            tmp_player *= (-1)
            node

        # print(tmp_mat)
        # print(f'win player {pl}')
        self.back_propagation(node, pl)

    def take_action(self, node):
        win_rate = -1
        best_action = []
        best_child = node.child[0]
        for child_node in node.child:
            win_rate_tmp = child_node.win_count / child_node.visited_count if child_node.visited_count != 0 else 0
            if win_rate_tmp > win_rate:
                best_action = child_node.action
                best_child = child_node

        return best_action, best_child

    def back_propagation(self, node, win_player):

        while node.parent:
            node.visited_count += 1
            if win_player == node.player:
                node.win_count += 1
            node = node.parent

    def UCT_cal(self, node, total_count):
        c = 2

        if (node.visited_count == 0) :
            utc_vl = float('inf')
        else:
            p_a = node.win_count / node.visited_count
            p_b = np.sqrt(c * np.log(total_count) / node.visited_count)
            utc_vl = p_a + p_b

        return utc_vl

    @staticmethod
    def win_condition(t_mat):

        width, height = t_mat.shape[0], t_mat.shape[1]

        for i in range(0, width):
            for j in range(0, height):
                player = int(t_mat[i][j])

                if player != 0:
                    if i <= width - 1 - 4:
                        chess_set = set(t_mat[i+k][j] for k in range(0,5))
                        # 横右向
                        if len(chess_set) == 1:
                            return player, True
                    if j <= height - 1 - 4:
                        # 纵下向
                        chess_set = set(t_mat[i][j+k] for k in range(0, 5))
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
    t_mat = np.zeros((8, 8))
    t_mat[3, 5] = 1
    # t_mat[3, 4] = 1
    # t_mat[4, 4] = 1
    # t_mat[5, 4] = 1
    # t_mat[6, 4] = 1
    # t_mat[7, 4] = 1

    # t_mat[2, 3] = 1
    # t_mat[3, 4] = 1
    # t_mat[4, 5] = 1
    # t_mat[5, 6] = 1
    # t_mat[6, 7] = 1

    # print(t_mat)
    node_1 = Node(t_mat, False, False, 1, True)
    tree = MentecarloTreeSearch(node_1, 8, t_mat)
    a,b=tree.run(node_1)
    # tree.choose_node(node_1)
    # print(node_1.state)
    # win condition test
    # instance = MentecarloTreeSearch(node_1, 8, t_mat)
    # player, win = instance.win_condition(t_mat)
    # print(player, win)
    # print(a)
    # print(b)
    # t_mat[a[0],a[1]] = b
    # print(t_mat)
    for child_s in node_1.child:
        print(f'{child_s.visited_count} and {child_s.win_count}')