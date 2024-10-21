import numpy as np

from mcts.mcts import MCTS
from mcts.node import TwoPlayerNode
from games.tictactoe import TicTacToeGameState, TicTacToeMove

# define inital state
init_board = np.zeros((7, 7))
init_state = TicTacToeGameState(board=init_board, turn="P1", win=4)

print(init_state)

# we keep this tree throughout the game
mcts = MCTS()
node = TwoPlayerNode(init_state, parent=None)

# keep playing until game terminates
while not node.state.is_terminal():
    # train for number of iterations
    for _ in range(10):
        mcts.train(node)
    # calculate best move, choose a move in the game, go one level down basically
    node = MCTS.choose(node)

    print(node.state)
    node.display()

# print result
print("Result of Game is: ", node.state.get_result()[0])
