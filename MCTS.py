def monte_carlo_tree_search(root):
    # while resources_left(time, computational power):
    #     leaf = traverse(root) # leaf = unvisited node 
    #     simulation_result = rollout(leaf)
    #     backpropagate(leaf, simulation_result)
    # return best_child(root)

    pass

# For the traverse function, to avoid using up too much time or resources, you may start considering only 
# a subset of children (e.g 5 children). Increase this number or by choosing this subset smartly later.
def traverse(node):
    # while fully_expanded(node):
    #     node = best_ucb(node)
    # return pick_univisted(node.children) or node # in case no children are present / node is terminal 

    pass
                                                 
def rollout(node):
    # while non_terminal(node):
    #     node = rollout_policy(node)
    # return result(node) 

    pass

def rollout_policy(node):
    # return pick_random(node.children)

    pass

def backpropagate(node, result):
    # if is_root(node) return 
    # node.stats = update_stats(node, result) 
    # backpropagate(node.parent)
    pass

def best_child(node):
    # pick child with highest number of visits
    pass