from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

"""
@classmethod has access to the class and can access or modify class state.
@staticmethod doesn't have access to the class or its state. still can use other static/class methods in the same class however.
"""


class Action(ABC):
    """abstract class for an action in the environment"""

    pass


class GameState(ABC):
    """
    This represents a distinct game state, not the overall environment!
    The overall environment is handled by Node
    """

    @abstractmethod
    # this is a forward reference to the class itself
    def act(self, action: Action) -> "GameState":
        """
        consumes action and returns another state of type GameState

        NOTE: only in act, do we flip the turn of the game. In no other functions should we actually advance the state of the game.
        """
        pass

    @abstractmethod
    def get_result(self) -> Optional[Tuple[str, int]]:
        """
        returns the final game result as a tuple of (Outcome, Reward), like ("P1",2)
        returns None if the game isnt over
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        returns boolean indicating if the game is over
        """
        pass

    @abstractmethod
    def get_legal_actions(self) -> List[Action]:
        """
        returns list of legal action at current game state
        """
        pass

    @abstractmethod
    def get_turn(self) -> str:
        """
        returns the player whose turn it is to act next
        """
        pass

    @staticmethod
    def other(turn) -> str:
        """
        returns the other player's name, as a string
        static method of the GameState.
        """
        return "P2" if turn == "P1" else "P1"
