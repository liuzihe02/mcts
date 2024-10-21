import numpy as np

from mcts.mcts import MCTS
from mcts.node import TwoPlayerNode
from games.tictactoe import TicTacToeGameState, TicTacToeMove


def play(board_size, win_cond, train_iterations, train_from_root: bool):
    """
    You play with a system trained using MCTS
    system's strategy is to use pure exploitation of the trained MCTS game tree.

    Args:
        board_size (int): width or height of the 2D board size
        win_cond (int): the number of elems in a row/col/diagonal to hit to fulfill the win condition
        train_iterations (int): at every chosen action, how many iterations to train in MCTS
        from_root (bool): at every chosen action, do I train from the root node of the game tree, or from the current node of the game.
    """
    # define inital state
    init_board = np.zeros((board_size, board_size))
    init_state = TicTacToeGameState(board=init_board, turn="P1", win=win_cond)

    print("=== TicTacToe Game===")
    print("Note that the (row,col) of (1,1) position is at the bottom left.")
    print(init_state)

    # we keep this tree throughout the game
    root = TwoPlayerNode(init_state, parent=None)
    cur_node = root

    # keep playing until game terminates
    while True:
        # check if system has won
        if cur_node.state.is_terminal():
            # system has won
            break

        # user inputs in the action
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        # take care of python indexing
        row -= 1
        col -= 1
        # move the board state and create a dummy node
        # the user is always the first to act, so takes the perspective of player 1 always
        new_user_state = cur_node.state.act(
            TicTacToeMove(row, col, "P1", TicTacToeGameState.P2V["P1"])
        )
        # update the current node that is indeed part of the game tree
        # note that the stats for this randomly chosen node will not be updated via backprop
        if train_from_root:
            cur_node = TwoPlayerNode(state=new_user_state, parent=None)
        else:
            # update current node, if we train from this node, this must be part of the game tree
            cur_node = TwoPlayerNode(state=new_user_state, parent=cur_node)

        print(cur_node.state)

        # check if user has won
        if cur_node.state.is_terminal():
            # user has won
            break

        # user has not won
        # train for number of iterations
        for _ in range(train_iterations):
            if train_from_root:
                # train all the way from the original root
                MCTS.train(root)
            else:
                # just train from this node
                MCTS.train(cur_node)

        # check if game has ended
        cur_node = MCTS.choose(cur_node)

        print("Opponent's Move:")
        print(cur_node.state)

    # print result
    print("Result of Game is: ", cur_node.state.get_result()[0])  # type: ignore #here, get_result should always return the tuple
    print("Stats from Root is", root.stats)


# needs about 1000000 iterations per move to be near-optimal
play(3, 3, 1000000, True)
