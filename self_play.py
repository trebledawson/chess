"""
Self-play arena for Brownie24.
"""
import numpy as np
import chess
import time
from random import randint
from keras.models import load_model
from linked_mcts import mcts
from tools import all_possible_moves, SearchTree

# This self-play function is used to generate training data
def self_play():
    game_start = time.time()
    model1 = load_model(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_live.h5')
    model2 = load_model(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_live.h5')

    # Initialize board and begin recording features for each board state
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    game_record = [[board], []]
    p1tree = SearchTree()
    p2tree = SearchTree()
    move = 1
    T = 1

    # Play game and record board state features for each move
    print('Game start.')
    while True:
        # Determine temperature coefficient by game length
        if move > 10:
            T = 0.1

        # Player 1 move
        print('Player 1 is thinking...')
        p1move, pi, p1tree, index = mcts(board, model1, poss_moves, T=T,
                                         tree=p1tree)
        board.push(chess.Move.from_uci(p1move))
        game_record[0].append(board)
        game_record[1].append(pi)

        print(board)
        print('Player 1: ', move, '. ', p1move, '\n', sep='')

        if board.is_game_over():
            winner = 0
            print('Winner: White.')
            print('Game duration:', time.time() - game_start, 'seconds. \n')
            break

        if move != 1:
            p2tree = p2tree.nodes[index]

        # Player 2 move
        print('Player 2 is thinking...')
        p2move, pi, p2tree, index = mcts(board, model2, poss_moves, T=T,
                                         tree=p2tree)
        board.push(chess.Move.from_uci(p2move))
        game_record[0].append(board)
        game_record[1].append(pi)

        print(board)
        print('Player 2: ', move, '... ', p2move, '\n', sep='')

        if board.is_game_over():
            winner = 1
            print('Winner: Black.')
            print('Game duration:', time.time() - game_start, 'seconds.\n')
            break

        p1tree = p1tree.nodes[index]

        move += 1

    del model1
    del model2
    del game_record[0][-1]

    if winner == 0:
        Z = np.empty((len(game_record[0]),1))
        Z[::2] = 1
        Z[1::2] = -1
        game_record.append(Z)

    else:
        Z = np.empty((len(game_record[0]), 1))
        Z[::2] = -1
        Z[1::2] = 1
        game_record.append(Z)

    return game_record

# This self-play function is used to determine the new generator model
def evaluation():
    train_color = randint(0, 1)
    if train_color == 0:
        model1 = load_model(filepath='C:\Glenn\Stuff\Machine '
                            'Learning\chess\models\model_train.h5')
        model2 = load_model(filepath='C:\Glenn\Stuff\Machine '
                            'Learning\chess\models\model_live.h5')
    else:
        model1 = load_model(filepath='C:\Glenn\Stuff\Machine '
                            'Learning\chess\models\model_live.h5')
        model2 = load_model(filepath='C:\Glenn\Stuff\Machine '
                            'Learning\chess\models\model_train.h5')

    # Initialize board and begin recording features for each board state
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    T = 0.01

    # Play game
    while True:
        # Player 1 move
        print('Player 1 is thinking...')
        p1move, _ = mcts(board, model1, poss_moves, T=T)
        board.push(chess.Move.from_uci(p1move))

        if board.is_game_over():
            winner = 0
            break

        # Player 2 move
        print('Player 2 is thinking...')
        p2move, _ = mcts(board, model2, poss_moves, T=T)
        board.push(chess.Move.from_uci(p2move))

        if board.is_game_over():
            winner = 1
            break

    del model1
    del model2

    if winner == train_color:
        return 1
    else:
        return 0
