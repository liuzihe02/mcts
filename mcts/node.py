from games.game import GameState, Action
from graphviz import Digraph
from typing import Dict


class Node:
    """
    Represents a node in the MCTS game tree.
    This also represents a single board state, along with relevant statistics tracking win/lose rewards for different players.

    This node contains the board class, which contains info on whose turn it is to move
    """

    def __init__(self, state: GameState, parent=None):
        """
        Initializes a new node.

        Args:
            state (GameState): The state of the game, that this node represents
            parent (Node): The parent node (None for the root).
        """
        self.state = state
        self.parent = parent
        # these are all the children nodes that have been explored, and are in the game tree
        # each time we expand on a leaf node, we consume an action from the unexplored actions, and add the new board state to children
        self.children = []
        # Number of times this node has been visited, by any player so far
        self.visits = 0

        # this is a dictionary, where each key-value corresponds to outcome-number
        # example, if there are 2 players, this would be {p1_wins:2,p2_wins:3,draws:4}

        # MUST BE IMPLEMENTED BY CHILD CLASS
        self.stats: Dict[str, float] = {}
        # at initialization, all these new actions are not in the game tree
        # add all the possible actions into unexplored_actions
        self.unexplored_actions = self.state.get_legal_actions()

    def get_unexplored_action(self) -> Action:
        """
        Selects and removes an unexplored action from the list of unexplored action

        Assumes there are unexplored actions!

        Returns:
            Action: the last unexplored action from the list of possible actions. Returns None if no actions left.
        """
        # assert len(self.unexplored_actions) > 0
        return self.unexplored_actions.pop()

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
        wins = self.stats[self.state.get_turn()]
        # loses from the perspective of the first player
        loses = self.stats[GameState.other(self.state.get_turn())]
        draws = self.stats["Draw"]
        return wins - loses - draws

    @property
    def N(self):
        return self.visits

    def display(self):
        dot = Digraph(comment="MCTS Tree")
        self._add_nodes_to_graph(dot)
        # save a different image each time you call this
        # can choose either svg for high quality, or png to view simple small trees
        dot.render("mcts_tree", view=False, cleanup=True, format="png")

    def _add_nodes_to_graph(self, dot: Digraph, parent_id=None):
        node_id = str(id(self))
        label = f"Visits: {self.visits} \n Stats: {self.stats} \nState:\n{self.state} \nTo Move:{self.state.get_turn()}"
        dot.node(node_id, label)

        if parent_id:
            dot.edge(parent_id, node_id)

        for child in self.children:
            child._add_nodes_to_graph(dot, node_id)
