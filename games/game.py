from abc import ABC, abstractmethod
from typing import List

"""
@classmethod has access to the class and can access or modify class state.
@staticmethod doesn't have access to the class or its state. still can use other static/class methods in the same class however.
"""


class Move(ABC):
    pass


class GameState(ABC):
    """this represents a distinct game state, not the overall environment!"""

    @abstractmethod
    def get_result(self):
        """
        the final game result as a tuple of (Outcome, Reward)
        like ("P1",2)

        returns None if the game isnt over
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        boolean indicating if the game is over,
        simplest implementation may just be

        Returns
        -------
        boolean

        """
        pass

    @abstractmethod
    # this is a forward reference to the class itself
    def move(self, action) -> "GameState":
        """
        consumes action and returns another state of type Game

        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[Move]:
        """
        returns list of legal action at current game state
        Returns
        -------
        list of AbstractGameAction

        """

    @staticmethod
    def other(turn):
        """returns the other player's name, as a string

        class method of the GameState

        Args:
            turn (str): a string of the other player
        """
        return "P2" if turn == "P1" else "P1"
