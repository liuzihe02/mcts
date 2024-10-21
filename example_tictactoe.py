import numpy as np

from mcts.mcts import MCTS
from mcts.node import TwoPlayerNode
from games.tictactoe import TicTacToeGameState, TicTacToeMove

# define inital state
init_board = np.zeros((3, 3))
init_state = TicTacToeGameState(board=init_board, turn="P1", win=3)

print(init_state)

# we keep this tree throughout the game
mcts = MCTS()
root = TwoPlayerNode(init_state, parent=None)
cur_node = root


# keep playing until game terminates
while not cur_node.state.is_terminal():
    # display only the entire game tree, from the root
    root.display()

    # train for number of iterations
    for _ in range(5):
        mcts.train(cur_node)
    # calculate best move, choose a move in the game, go one level down basically
    cur_node = MCTS.choose(cur_node)

    print(cur_node.state)


# print result
print("Result of Game is: ", node.state.get_result()[0])
