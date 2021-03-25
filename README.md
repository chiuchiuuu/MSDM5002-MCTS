# MSDM5002-MCTS

An gomoku AI using MCTS algorithm.

## How to run


```shell
python gomoku.py
```

parameters:

```python
# human vs MCTS
gomoku  = Gomoku(8, HumanPlayer(), MCTSPlayer(max_time=10))
# human vs MCTS with policy network
gomoku = Gomoku(8, HumanPlayer(), MCTSPlayerAlpha(policy_value_fn,max_time=10))
```