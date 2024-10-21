import numpy as np

# look in the same directory as current one
from .node import Node, TwoPlayerNode
from games.game import Action
from typing import Tuple


class MCTS:
    """
    Implementation of the Monte Carlo Tree Search algorithm.
    This class does not store anything at all! Only provides static methods for interacting with Nodes
    All information is stored in the nodes and the relationship between them

    Currently ONLY supports 2 player version.
    """

    @staticmethod
    def _select(root: Node) -> Node:
        """
        Selects a leaf node in the whole game tree to do the expansion step.

        This method implements the selection step of MCTS, traversing the tree from the root to a leaf node using the UCT formula.
        This node is allowed to be a terminal node, in which case no simulation will occur from this node, but backprop still occurs.

        Args:
            root (Node): the root node of the game tree, that we want to select one of the leaf nodes in this game tree

        Returns:
            Node: The selected leaf node, for rollout
        """
        node = root
        # handle case of a terminal node (game has terminated)
        while not node.state.is_terminal():
            # while this node still has actions that are not in the game tree
            if len(node.unexplored_actions) > 0:
                # this node will be selected for simulation/rollout
                return node
            # all nodes have already been explored, go one level deeper
            else:
                # descend one layer deeper, with some exploration
                # this is the tree policy, different from rollout policy
                node = MCTS._UCT(node, c_explore=1.4)

        # handle terminal node
        return node

    @staticmethod
    def _expand(node: Node) -> Node:
        """
        Expands the game tree, by adding a child node to the given leaf node with an untried action.

        NOTE: Assumes the given node still has unexplored actions

        Args:
            node (Node): The node to expand on.

        Returns:
            Node: The newly created child node
        """
        assert len(node.unexplored_actions) > 0
        # updates this node's unexplored actions too
        action = node.get_unexplored_action()
        # advance to the next state
        next_state = node.state.act(action)
        # make this next state for the child node
        child = TwoPlayerNode(next_state, parent=node)
        node.children.append(child)
        return child

    @staticmethod
    def _simulate(node: Node) -> Tuple[str, float]:
        """
        Returns the results for a random simulation (to completion) of the given node
        Only a single round of simulation.
        NOTE: Allows the node to be terminal, in which case the result is immediately returned

        Args:
            node (Node): the given node to start simulation from

        Returns:
            Tuple[str, float]: the tuple containing the result of simulation
        """

        cur_state = node.state
        while not cur_state.is_terminal():
            # get all the possible moves
            actions = cur_state.get_legal_actions()
            action = MCTS._rollout_policy(actions)
            # this takes care of the swapping of players
            cur_state = cur_state.act(action)

        # the result of the ending condition of the game. this is a string containing the outcome, followed by the reward
        return cur_state.get_result()  # type: ignore

    @staticmethod
    def _backpropagate(node: Node, result: Tuple[str, float]):
        """
        Backpropagates the reward value up the tree, updating node statistics.
        Updates the stats for this node, and recurses this function on its parent

        Args:
            node (Node): the end node to do backprop from
            result (Tuple[str, int]): the results of the simulation
        """
        node.visits += 1

        if result[0] == "P1":
            node.stats["P1"] += result[1]

        elif result[0] == "P2":
            node.stats["P2"] += result[1]

        elif result[0] == "Draw":
            node.stats["Draw"] += result[1]

        if node.parent:
            MCTS._backpropagate(node.parent, result)

    @staticmethod
    def _rollout_policy(actions) -> Action:
        """returns a random action from the available choice of actions"""
        return actions[np.random.randint(len(actions))]

    @staticmethod
    def _UCT(node: Node, c_explore: float) -> Node:
        """UCB for trees (one layer down only)"""
        choices = [
            (c.Q / c.N) + c_explore * np.sqrt((2 * np.log(node.N) / c.N))
            for c in node.children
        ]
        return node.children[np.argmax(choices)]

    @staticmethod
    def choose(node: Node) -> Node:
        """
        Choose the best successor of node. (Choose a move while playing game)
        NOTE: This node be a terminal node.
        """
        if node.state.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        # no children in the game tree, just use rollout policy
        if len(node.children) == 0:
            actions = node.state.get_legal_actions()
            # basically random choice
            action = MCTS._rollout_policy(actions)
            # this takes care of the swapping of players
            new_state = node.state.act(action)
            # return a dummy node that is not part of the game tree
            # note that the stats for this randomly chosen node will not be updated via backprop
            return TwoPlayerNode(state=new_state, parent=None)

        # exploitation only
        return MCTS._UCT(node, c_explore=0)

    @staticmethod
    def train(root: Node):
        """
        Does one iteration of Monte Carlo Tree Search with the 4 core steps.
        Essentially adds one more child to the game tree and do backprop on the tree.
        Does training only from the root node provided!

        Args:
            root (Node): the root of the existing game tree
        """

        leaf = MCTS._select(root)

        # this node doesnt represent an end state
        if not leaf.state.is_terminal():
            # must have unexplored actions, so we can create child Nodes
            assert len(leaf.unexplored_actions) > 0
            # new child in the game tree
            child = MCTS._expand(leaf)
            # simulation results from this child
            result = MCTS._simulate(child)
            # now we backprop the empty child, which currently has no stats/simulation in it
            MCTS._backpropagate(child, result)
        # the leaf node is in fact terminal
        else:
            # no point expanding, get results directly
            result = MCTS._simulate(leaf)
            MCTS._backpropagate(leaf, result)
