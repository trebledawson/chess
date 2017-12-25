"""
Toolbox of helper functions:
----------
The SearchTree class is a decision tree that is iteratively generated during
the MCTS algorithm.
----------
The features() function accepts a chess.Bitboard() object and returns:
-(8x8x1) numpy array of white piece locations
-(8x8x1) numpy array of black piece locations
-(1x1) numpy array of current player information
---------
The all_possible_moves() function returns a list of all 1968 possible moves
on a chessboard in UCI notation in alphabetical order.
---------
The get_move() function is used to select a move based on DeepMind's modified
PUCT equation.
---------
The get_pi() function is used to obtain the probability distribution of each
legal move in a given board state.
"""

import numpy as np
import chess
from math import sqrt

class SearchTree:
    def __init__(self, **kwargs):
        N = kwargs.get('N', 0)
        W = kwargs.get('W', 0)
        Q = kwargs.get('Q', 0)
        P = kwargs.get('P', 0)
        name = kwargs.get('name', 0000)
        self.nodes = []
        self.data = [N, W, Q, P]
        self.name = name

    def create_node(self, **kwargs):
        P = kwargs.get('P', 0)
        name = kwargs.get('name', 0000)
        self.nodes.append(SearchTree(P=P, name=name))

def features(boardfen):
    fen = boardfen.split()

    # Self-contained type checker
    def int_if_int(s):
        try:
            int(s)
            return int(s)
        except ValueError:
            return s

    # Convert board to piece arrays
    wp = np.zeros(64).astype('float32')
    wn = np.zeros(64).astype('float32')
    wb = np.zeros(64).astype('float32')
    wr = np.zeros(64).astype('float32')
    wq = np.zeros(64).astype('float32')
    wk = np.zeros(64).astype('float32')
    bp = np.zeros(64).astype('float32')
    bn = np.zeros(64).astype('float32')
    bb = np.zeros(64).astype('float32')
    br = np.zeros(64).astype('float32')
    bq = np.zeros(64).astype('float32')
    bk = np.zeros(64).astype('float32')
    empty = np.zeros(64).astype('float32')
    count = 0
    for letter in fen[0]:
        letter = int_if_int(letter)
        if letter in range(9):
            while letter > 0:
                empty[count] = 1
                letter -= 1
                count += 1
        elif letter == 'p':
            bp[count] = 1
            count += 1
        elif letter == 'P':
            wp[count] = 1
            count += 1
        elif letter == '/':
            continue
        elif letter == 'b':
            bb[count] = 1
            count += 1
        elif letter == 'B':
            wb[count] = 1
            count += 1
        elif letter == 'n':
            bn[count] = 1
            count += 1
        elif letter == 'N':
            wn[count] = 1
            count += 1
        elif letter == 'r':
            br[count] = 1
            count += 1
        elif letter == 'R':
            wr[count] = 1
            count += 1
        elif letter == 'q':
            bq[count] = 1
            count += 1
        elif letter == 'Q':
            wq[count] = 1
            count += 1
        elif letter == 'k':
            bk[count] = 1
            count += 1
        elif letter == 'K':
            wk[count] = 1
            count += 1
    wp = wp.reshape(1, 8, 8)
    wn = wn.reshape(1, 8, 8)
    wb = wb.reshape(1, 8, 8)
    wr = wr.reshape(1, 8, 8)
    wq = wq.reshape(1, 8, 8)
    wk = wk.reshape(1, 8, 8)
    bp = bp.reshape(1, 8, 8)
    bn = bn.reshape(1, 8, 8)
    bb = bb.reshape(1, 8, 8)
    br = br.reshape(1, 8, 8)
    bq = bq.reshape(1, 8, 8)
    bk = bk.reshape(1, 8, 8)
    empty = empty.reshape(1, 8, 8)

    # Determine current player
    player = np.zeros((1, 8, 8)).astype('float32')
    if fen[1] == 'b':
        player += 1

    features = np.concatenate((wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk,
                               empty, player), axis=0)

    return features

def all_possible_moves():
    moves = []
    board1 = chess.Bitboard()
    board2 = chess.Bitboard()
    board3 = chess.Bitboard()

    # WHITE #
    # Row 1
    fen1 = '8/8/8/8/8/8/8/Q7 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/N7 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves+list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/1Q6 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/1N6 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/2Q5 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/2N5 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/3Q4 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/3N4 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/4Q3 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/4N3 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/5Q2 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/5N2 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/6Q1 w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/6N1 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/7Q w - - 0 1'
    fen2 = '8/8/8/8/8/8/8/7N w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Row 2
    fen1 = '8/8/8/8/8/8/Q7/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/N7/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/1Q6/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/1N6/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/2Q5/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/2N5/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/3Q4/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/3N4/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/4Q3/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/4N3/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/5Q2/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/5N2/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/6Q1/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/6N1/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/7Q/8 w - - 0 1'
    fen2 = '8/8/8/8/8/8/7N/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Row 3
    fen1 = '8/8/8/8/8/Q7/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/N7/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/1Q6/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/1N6/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/2Q5/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/2N5/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/3Q4/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/3N4/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/4Q3/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/4N3/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/5Q2/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/5N2/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/6Q1/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/6N1/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/7Q/8/8 w - - 0 1'
    fen2 = '8/8/8/8/8/7N/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Row 4
    fen1 = '8/8/8/8/Q7/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/N7/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/1Q6/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/1N6/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/2Q5/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/2N5/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/3Q4/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/3N4/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/4Q3/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/4N3/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/5Q2/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/5N2/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/6Q1/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/6N1/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/7Q/8/8/8 w - - 0 1'
    fen2 = '8/8/8/8/7N/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Row 5
    fen1 = '8/8/8/Q7/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/N7/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/1Q6/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/1N6/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/2Q5/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/2N5/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/3Q4/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/3N4/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/4Q3/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/4N3/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/5Q2/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/5N2/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/6Q1/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/6N1/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/7Q/8/8/8/8 w - - 0 1'
    fen2 = '8/8/8/7N/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Row 6
    fen1 = '8/8/Q7/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/N7/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/1Q6/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/1N6/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/2Q5/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/2N5/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/3Q4/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/3N4/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/4Q3/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/4N3/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/5Q2/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/5N2/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/6Q1/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/6N1/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/7Q/8/8/8/8/8 w - - 0 1'
    fen2 = '8/8/7N/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Row 7
    fen1 = '8/Q7/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/N7/8/8/8/8/8/8 w - - 0 1'
    fen3 = '1nnnnnnn/P7/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/1Q6/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/1N6/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'n1nnnnnn/1P6/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/2Q5/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/2N5/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'nn1nnnnn/2P5/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/3Q4/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/3N4/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'nnn1nnnn/3P4/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/4Q3/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/4N3/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'nnnn1nnn/4P3/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/5Q2/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/5N2/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'nnnnn1nn/5P2/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/6Q1/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/6N1/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'nnnnnn1n/6P1/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/7Q/8/8/8/8/8/8 w - - 0 1'
    fen2 = '8/7N/8/8/8/8/8/8 w - - 0 1'
    fen3 = 'nnnnnnn1/7P/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))


    # Row 8
    fen1 = 'Q7/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = 'N7/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '1Q6/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '1N6/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '2Q5/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '2N5/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '3Q4/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '3N4/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '4Q3/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '4N3/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '5Q2/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '5N2/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '6Q1/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '6N1/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '7Q/8/8/8/8/8/8/8 w - - 0 1'
    fen2 = '7N/8/8/8/8/8/8/8 w - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # BLACK #
    # Rob 1
    fen1 = '8/8/8/8/8/8/8/q7 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/n7 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/1q6 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/1n6 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/2q5 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/2n5 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/3q4 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/3n4 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/4q3 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/4n3 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/5q2 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/5n2 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/6q1 b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/6n1 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/8/8/7q b - - 0 1'
    fen2 = '8/8/8/8/8/8/8/7n b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Rob 2
    fen1 = '8/8/8/8/8/8/q7/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/n7/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/p7/1NNNNNNN b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/1q6/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/1n6/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/1p6/N1NNNNNN b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/2q5/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/2n5/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/2p5/NN1NNNNN b - - 0 1'

    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/3q4/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/3n4/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/3p4/NNN1NNNN b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/4q3/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/4n3/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/4p3/NNNN1NNN b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/5q2/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/5n2/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/5p2/NNNNN1NN b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/6q1/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/6n1/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/6p1/NNNNNN1N b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    fen1 = '8/8/8/8/8/8/7q/8 b - - 0 1'
    fen2 = '8/8/8/8/8/8/7n/8 b - - 0 1'
    fen3 = '8/8/8/8/8/8/7p/NNNNNNN1 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    board3.set_fen(fen3)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    legal3 = [move.uci() for move in board3.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2 + legal3))))

    # Rob 3
    fen1 = '8/8/8/8/8/q7/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/n7/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/1q6/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/1n6/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/2q5/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/2n5/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/3q4/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/3n4/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/4q3/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/4n3/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/5q2/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/5n2/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/6q1/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/6n1/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/8/7q/8/8 b - - 0 1'
    fen2 = '8/8/8/8/8/7n/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Rob 4
    fen1 = '8/8/8/8/q7/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/n7/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/1q6/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/1n6/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/2q5/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/2n5/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/3q4/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/3n4/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/4q3/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/4n3/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/5q2/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/5n2/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/6q1/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/6n1/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/8/7q/8/8/8 b - - 0 1'
    fen2 = '8/8/8/8/7n/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Rob 5
    fen1 = '8/8/8/q7/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/n7/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/1q6/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/1n6/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/2q5/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/2n5/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/3q4/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/3n4/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/4q3/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/4n3/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/5q2/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/5n2/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/6q1/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/6n1/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/8/7q/8/8/8/8 b - - 0 1'
    fen2 = '8/8/8/7n/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Rob 6
    fen1 = '8/8/q7/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/n7/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/1q6/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/1n6/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/2q5/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/2n5/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/3q4/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/3n4/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/4q3/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/4n3/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/5q2/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/5n2/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/6q1/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/6n1/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/8/7q/8/8/8/8/8 b - - 0 1'
    fen2 = '8/8/7n/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Rob 7
    fen1 = '8/q7/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/n7/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/1q6/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/1n6/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/2q5/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/2n5/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/3q4/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/3n4/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/4q3/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/4n3/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/5q2/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/5n2/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/6q1/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/6n1/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '8/7q/8/8/8/8/8/8 b - - 0 1'
    fen2 = '8/7n/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    # Rob 8
    fen1 = 'q7/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = 'n7/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '1q6/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '1n6/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '2q5/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '2n5/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '3q4/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '3n4/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '4q3/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '4n3/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '5q2/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '5n2/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '6q1/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '6n1/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    fen1 = '7q/8/8/8/8/8/8/8 b - - 0 1'
    fen2 = '7n/8/8/8/8/8/8/8 b - - 0 1'
    board1.set_fen(fen1)
    board2.set_fen(fen2)
    legal1 = [move.uci() for move in board1.generate_legal_moves(king=False)]
    legal2 = [move.uci() for move in board2.generate_legal_moves(king=False)]
    moves = list(set(moves + list(set(legal1 + legal2))))

    moves.sort()
    return moves

def get_move(edges, C):
    edges = np.array(edges)
    U = np.zeros(len(edges))
    for move in range(len(edges)):
        frac = (sqrt(np.sum(edges[:, 0]))) / (1 + edges[move, 0])
        P = edges[move, 3]
        U[move] = C * P * frac

    a = edges[:, 2] + U
    return np.random.choice(np.flatnonzero(a == a.max()))

def get_pi(visits, T):
    pow = 1 / T
    visits = np.array(np.ravel(visits))
    pi = np.zeros(len(visits))
    for move in range(len(visits)):
        pi_m = (visits[move] ** pow) / np.sum(visits ** pow)
        if pi_m == np.inf:
            pi_m = 1
        pi[move] = pi_m

    return pi
