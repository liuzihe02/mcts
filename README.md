# Monte Carlo Tree Search
This is a lightweight implementation of Monte Carlo Tree Search (MCTS) for 2 player games. This repo is heavily inspired by [int8's repo](https://github.com/int8/monte-carlo-tree-search) and [qpwo's repo](https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1), and captures elements from both.

## Features

- Visualization of game tree
- Support for both auto-play and human-VS-AI modes
- Easy extension to new environments via `Game` class

## Structure

- `mcts/`
    - `mcts.py` contains core algorithims of MCTS
    - `node.py` contains the class for a node in the game tree

- `games/`
    - `game.py` defines abstract base classes for games and actions
    - `tictactoe.py` implements the tic tac toe game

## Usage

- `example_ttt_autoplay.py` simulates 2 AI playing against each other, training using MCTS in real time
- `example_TTT_play.py` allows you to interact with the system opponent with MCTS

Note that training for many iterations (>10000) will lead to optimal play from both sides, hence the root node will contain many more draws than wins from either Player 1 or Player 2.

## Theory

We borrow notes and figures from [int8's website](https://int8.io/monte-carlo-tree-search-beginners-guide/) and the [MCTS Wikipedia page](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search). Refer to `theory/theory.md` for detailed explanations.











