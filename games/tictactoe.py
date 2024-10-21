import numpy as np
from .game import GameState, Move


class TicTacToeMove(Move):
    def __init__(self, x_coord, y_coord, turn, value):
        self.x_coord = x_coord
        self.y_coord = y_coord
        # whose turn it is to play
        self.turn = turn
        # represents what you want to put in the board, will be an integer here
        self.value = value

    def __repr__(self):
        return "x:{0} y:{1} turn:{2} value:{3}".format(
            self.x_coord, self.y_coord, self.turn, self.value
        )


class TicTacToeGameState(GameState):
    """
    currently only supports 2 player implementation

    NOTE:
    Player 1 - represented by X - value of 1
    Player 2 - represented by O - value of -1
    Empty Squares - value of 0

    Player 1 will always START FIRST!
    """

    # these are class variables
    # maps player to symbol and value to put on the board
    P2V = {"P1": 1, "P2": -1}
    # maps value on board to the actual symbol
    V2S = {1: "X", -1: "O", 0: " "}

    def __init__(self, board: np.ndarray, win: int, turn: str):
        super().__init__()
        # check validity of board
        if len(board.shape) != 2 or board.shape[0] != board.shape[1]:
            raise ValueError("Only 2D square boards allowed")

        # this is a numpy array
        self.board = board
        # this is an integer indicating width/height of board
        self.board_size = board.shape[0]
        # this is the number of patches need to win, the win condition
        assert win <= self.board_size
        self.win = win
        # the turn here represent which player it is to MOVE NEXT
        self.turn = turn

    def get_result(self):
        """
        return None if the game is not over yet
        """

        """
        This loop checks for wins in rows and columns.
        It uses a sliding window of size self.win across rows and columns.
        rowsum sums self.win consecutive elements in each row.
        colsum sums self.win consecutive elements in each column.
        If any sum equals self.win, player X wins (assuming X is represented by 1).
        If any sum equals -self.win, player O wins (assuming O is represented by -1).
        """
        for i in range(self.board_size - self.win + 1):
            rowsum = np.sum(self.board[i : i + self.win], 0)
            colsum = np.sum(self.board[:, i : i + self.win], 1)
            if rowsum.max() == self.win or colsum.max() == self.win:
                return ("P1", 1)
            if rowsum.min() == -self.win or colsum.min() == -self.win:
                return ("P2", 1)
        """
        This nested loop checks for diagonal wins.
        It uses a sliding window of size self.win x self.win across the board.
        For each window:
        diag_sum_tl checks the top-left to bottom-right diagonal.
        diag_sum_tr checks the top-right to bottom-left diagonal.
        If any diagonal sum equals self.win or -self.win, the respective player wins.
        """
        for i in range(self.board_size - self.win + 1):
            for j in range(self.board_size - self.win + 1):
                sub = self.board[i : i + self.win, j : j + self.win]
                diag_sum_tl = sub.trace()
                diag_sum_tr = sub[::-1].trace()
                if diag_sum_tl == self.win or diag_sum_tr == self.win:
                    return ("P1", 1)
                if diag_sum_tl == -self.win or diag_sum_tr == -self.win:
                    return ("P2", 1)

        # draw
        if np.all(self.board != 0):
            # give 0.5 to both
            return ("Draw", 0.5)

        # if not over - no result
        return None

    def get_turn(self):
        return self.turn

    def is_terminal(self):
        """has the game ended yet"""

        return self.get_result() is not None

    def is_move_legal(self, move: TicTacToeMove):
        # check if correct player moves
        if move.turn != self.turn:
            return False

        # check if inside the board on x-axis
        x_in_range = 0 <= move.x_coord < self.board_size
        if not x_in_range:
            return False

        # check if inside the board on y-axis
        y_in_range = 0 <= move.y_coord < self.board_size
        if not y_in_range:
            return False

        # finally check if board field not occupied yet
        return self.board[move.x_coord, move.y_coord] == 0

    def move(self, move: TicTacToeMove):
        # move here already contains information about whose turn it is to move
        # check if legal
        if not self.is_move_legal(move):
            raise ValueError(
                "move {0} on board {1} is not legal".format(move, self.board)
            )
        # create a copy of the board
        new_board = np.copy(self.board)
        # update the board
        new_board[move.x_coord, move.y_coord] = move.value
        # FLIP to the opponent's turn, and return the new BoardState
        return TicTacToeGameState(new_board, self.win, GameState.other(self.turn))

    def get_legal_actions(self):
        indices = np.where(self.board == 0)
        return [
            TicTacToeMove(
                coords[0], coords[1], self.turn, TicTacToeGameState.P2V[self.turn]
            )
            for coords in list(zip(indices[0], indices[1]))
        ]

    def __repr__(self):
        def stringify(row):
            return " " + " | ".join(map(lambda x: self.V2S[int(x)], row)) + " "

        board = self.board.copy().T[::-1]
        rows = [stringify(row) for row in board]
        separator = "\n" + "-" * (len(board[0]) * 4 - 1) + "\n"

        return separator.join(rows) + "\n\n"
