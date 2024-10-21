import random
import math
import uuid
import numpy as np
from games.game import GameState, Move
from typing import List
from graphviz import Digraph


class Node:
    """
    Represents a node in the MCTS game tree.
    These are nodes that already have been expanded on.
    This also represents a single board state, along with relevant statistics tracking win rates
    """

    def __init__(self, state: GameState, parent=None):
        """
        Initializes a new node.

        Args:
            state (Game): The game state at this node
            parent (Node): The parent node (None for the root).
        """
        self.state = state
        self.parent = parent
        # these are all the children nodes that have been explored, and are in the game tree
        self.children = []
        # Number of times this node has been visited, by any player so far
        self.visits = 0

        # this is a dictionary, where each key-value corresponds to outcome-number
        # example, if there are 2 players, this would be {p1_wins:2,p2_wins:3,draws:4}
        # any new element will be initialized with the integer 0

        # MUST BE IMPLEMENTED BY CHILD CLASS
        self.stats = None
        # at initialization, all these new actions are not in the game tree
        self.unexplored_actions = self.state.get_legal_actions()

    def get_unexplored_action(self):
        """
        Selects and removes an unexplored action from the list of unexplored action

        Returns:
            Action: the last unexplored action from the list of possible actions. Returns None if no actions left.
        """
        return self.unexplored_actions.pop() if self.unexplored_actions else None

    @property
    def Q(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @property
    def N(self):
        raise NotImplementedError("Subclass must implement abstract method")


class TwoPlayerNode(Node):
    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.stats = {"P1": 0, "P2": 0, "Draw": 0}

    @property
    def Q(self):
        # ensure of whether choose parent state or my state
        # TODO: note that turn is not implemented in the abstract class
        wins = self.stats[self.state.turn]
        # loses from the perspective of the first player
        loses = self.stats[GameState.other(self.state.turn)]
        return wins - loses

    @property
    def N(self):
        return self.visits

    def display(self):
        dot = Digraph(comment="MCTS Tree")
        self._add_nodes_to_graph(dot)
        # save a different image each time you call this
        dot.render(
            f"mcts_tree{str(uuid.uuid4)}", view=False, cleanup=True, format="png"
        )

    def _add_nodes_to_graph(self, dot: Digraph, parent_id=None):
        node_id = str(id(self))
        label = f"Visits: {self.visits}\nStats: {self.stats}\nState: \n{self.state}"
        dot.node(node_id, label)

        if parent_id:
            dot.edge(parent_id, node_id)

        for child in self.children:
            child._add_nodes_to_graph(dot, node_id)
