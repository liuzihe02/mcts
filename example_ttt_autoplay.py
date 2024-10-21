import numpy as np

from mcts.mcts import MCTS
from mcts.node import TwoPlayerNode
from games.tictactoe import TicTacToeGameState


def autoplay(
    board_size: int,
    win_cond: int,
    train_iterations: int,
    train_from_root: bool,
    display: bool,
):
    """
    runs a simulation game of 2 players tic tac toe.
    strategy is to use pure exploitation of the trained MCTS game tree.

    Args:
        board_size (int): width or height of the 2D board size
        win_cond (int): the number of elems in a row/col/diagonal to hit to fulfill the win condition
        train_iterations (int): at every chosen action, how many iterations to train in MCTS
        from_root (bool): at every chosen action, do I do the training from the root node of the game tree, or from the current node of the game.
        display (bool): whether or not to display the entire game tree in the end, as a png. Not reccomended for huge game trees.
    """
    # define inital state
    init_board = np.zeros((board_size, board_size))
    init_state = TicTacToeGameState(board=init_board, turn="P1", win=win_cond)

    print(init_state)

    # we keep this tree throughout the game
    root = TwoPlayerNode(init_state, parent=None)
    cur_node = root

    # keep playing until game terminates
    while not cur_node.state.is_terminal():
        # # display the entire game tree, from the root. will be saved as a png
        # root.display()

        # train for number of iterations
        for _ in range(train_iterations):
            if train_from_root:
                # currently doesnt work if you train from root, only works if you train from cur_node
                MCTS.train(root)
            else:
                MCTS.train(cur_node)

        # calculate best move, choose a move in the game, go one level down basically
        cur_node = MCTS.choose(cur_node)

        print(cur_node.state)

    # print result
    print("Result of Game is: ", cur_node.state.get_result()[0])  # type: ignore #here, get_result should always return the tuple
    print("Stats from Root is", root.stats)

    if display:
        root.display()


autoplay(3, 3, 10, False, False)
