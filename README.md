# Othello - Python Project

A complete implementation of the strategy game Othello (Reversi) in Python, with a Tkinter graphical interface and an artificial intelligence based on the Minimax algorithm with Alpha-Beta pruning.

---

## Description

This project offers a playable version of the game Othello where a human player can compete against a competitive AI. The AI uses a dynamic evaluation of the board that adapts to the game phase (beginning, middle, end of game) to make strategic decisions.

---

## Features

- Intuitive graphical interface developed with Tkinter
- 8x8 board conforming to the official rules of Othello
- Competitive AI based on:
  - The Minimax algorithm (depth 5)
  - Alpha-Beta pruning to optimize calculations
  - A dynamic evaluation function (position, tokens, mobility)
  - A square importance table (corners, edges, center)
- Automatic detection of:
  - Valid moves
  - Token flips
  - End of game (no more possible moves or board full)
- Display of the winner or a draw at the end of the game

---

## Prerequisites

- Python 3.8+
- NumPy (for matrix management of the board)

### Installation

```bash
pip install numpy
