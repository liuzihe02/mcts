import random
import math
import numpy as np
from collections import defaultdict

# look in the same directory as current one
from .node import TwoPlayerNode
from games.game import GameState


class MCTS:
    """
    Implementation of the Monte Carlo Tree Search algorithm.

    does not store anything at all! all information is stored in the nodes and the relationship between them

    Currently ONLY supports 2 player version.
    """

    def _select(self, root):
        """
        Selects a leaf node in the whole game tree to do the expansion step.

        This method implements the selection step of MCTS, traversing the tree
        from the root to a leaf node using the UCT formula.

        NOTE: this method only selects a node from the root node given!

        Args:
            node (Node): The starting node for selection (usually the root).

        Returns:
            Node: The selected leaf node, for rollout
        """
        node = root
        # handle case of an actual terminal node
        while not node.state.is_terminal():
            # while this node still has actions that are not in the game tree
            if len(node.unexplored_actions) > 0:
                return self._expand(node)
            else:
                # descend one layer deeper, with some exploration
                # this is the tree policy, different from rollout policy
                node = MCTS._UCT(node, c_explore=1.4)

        # handle terminal node, or has no children
        return node

    def _expand(self, node: TwoPlayerNode):
        """
        Expands the given node by adding a child node with an untried move.

        This method implements the expansion step of MCTS.

        Args:
            node (Node): The node to expand.

        Returns:
            Node: The newly created child node, or the input node if fully expanded.
        """
        # updates this node too
        action = node.get_unexplored_action()
        # advance to the next state
        next_state = node.state.move(action)
        # make this next state for the child node
        child = TwoPlayerNode(next_state, parent=node)
        node.children.append(child)
        return child

    def _simulate(self, node: TwoPlayerNode):
        """Returns the reward for a random simulation (to completion) of `node`
        Assumed here this node has more unexplored children
        only a single round of simulation"""
        cur_state = node.state
        while not cur_state.is_terminal():
            # get all the possible moves
            moves = cur_state.get_legal_actions()
            action = MCTS._rollout_policy(moves)
            # this takes care of the swapping of players
            cur_state = cur_state.move(action)

        # the result of the ending condition of the game. this is a string containing the outcome, followed by the reward
        return cur_state.get_result()

    def _backpropagate(self, node, result):
        """
        Backpropagates the reward value up the tree, updating node statistics.

        Updates the stats for this node, and recurses this function on its parent

        This method implements the backpropagation step of MCTS.

        Args:
            node (Node): The starting node for backpropagation.
            reward (float): The reward value to backpropagate.
        """
        node.visits += 1

        if result[0] == "P1":
            node.stats["P1"] += result[1]

        elif result[0] == "P2":
            node.stats["P2"] += result[1]

        elif result[0] == "Draw":
            node.stats["P1"] += result[1]
            node.stats["P2"] += result[1]

        if node.parent:
            self._backpropagate(node.parent, result)

    @staticmethod
    def _rollout_policy(moves):
        """returns a random move from the available choice of moves"""
        return moves[np.random.randint(len(moves))]

    @staticmethod
    def _UCT(node: TwoPlayerNode, c_explore: float) -> TwoPlayerNode:
        """UCB for trees (one layer down only)"""
        choices = [
            (c.Q / c.N) + c_explore * np.sqrt((2 * np.log(node.N) / c.N))
            for c in node.children
        ]
        return node.children[np.argmax(choices)]

    @staticmethod
    def choose(node: TwoPlayerNode) -> TwoPlayerNode:
        "Choose the best successor of node. (Choose a move in the game)"
        if node.state.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        # no children in the game tree, just use rollout policy
        if len(node.children) == 0:
            moves = node.state.get_legal_actions()
            action = MCTS._rollout_policy(moves)
            return action

        # exploitation only
        return MCTS._UCT(node, c_explore=0)

    def train(self, root):
        """Make the tree one layer better. (Train for one iteration.)

        NOTE: this algo only trains from the root node provided!"""
        leaf = self._select(root)
        self._expand(leaf)
        result = self._simulate(leaf)
        self._backpropagate(leaf, result)
