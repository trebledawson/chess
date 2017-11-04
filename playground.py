"""
Playground for testing things
"""

import chess
from mcts import mcts
from tools import features


def main():
    # Example board for debugging
    board = chess.Bitboard()
    # print(board, '\n')
    whi, bla, turn, cas, enp = features(board)
    print(whi, '\n')
    print(bla, '\n')
    print(turn, '\n')
    print(cas, '\n')
    print(enp, '\n')


if __name__ == '__main__':
    main()