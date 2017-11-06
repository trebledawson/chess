"""
Self-play arena for Brownie24.
"""
import numpy as np
import chess
import time
from random import randint
from keras.models import load_model
from linked_mcts import mcts
from tools import features, all_possible_moves

def self_play():
    game_start = time.time()
    model1 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')
    model2 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')

    # Initialize board and begin recording features for each board state
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    w0, b0, p0, c0, ep0 = features(board)
    game_features = [[w0], [b0], [p0], [c0], [ep0], []]
    move = 1
    T = 1

    # Play game and record board state features for each move
    while True:
        if move > 10:
            T = 0.1
        # Player 1 move
        print('Player 1 is thinking...')
        p1move, pi = mcts(board, model1, poss_moves, T=T)
        board.push(chess.Move.from_uci(p1move))
        w, b, p, c, ep = features(board)
        game_features[0].append(w)
        game_features[1].append(b)
        game_features[2].append(p)
        game_features[3].append(c)
        game_features[4].append(ep)
        game_features[5].append(pi)

        print(board)
        print('Player 1 plays', p1move, '\n')

        if board.is_game_over():
            winner = 0
            print('Winner: White.')
            print('Game duration:', time.time() - game_start, 'seconds. \n')
            break

        # Player 2 move
        print('Player 2 is thinking...')
        p2move, pi = mcts(board, model2, poss_moves, T=T)
        board.push(chess.Move.from_uci(p2move))
        w, b, p, c, ep = features(board)
        game_features[0].append(w)
        game_features[1].append(b)
        game_features[2].append(p)
        game_features[3].append(c)
        game_features[4].append(ep)
        game_features[5].append(pi)

        print(board)
        print('Player 2 plays', p2move, '\n')

        if board.is_game_over():
            winner = 1
            print('Winner: Black.')
            print('Game duration:', time.time() - game_start, 'seconds.\n')
            break

        move += 1

    del model1
    del model2

    if winner == 0:
        Z = np.empty((len(game_features[2]),1))
        Z[::2] = 1
        Z[1::2] = -1
        game_features.append(Z)

    else:
        Z = np.empty((len(game_features[2]), 1))
        Z[::2] = -1
        Z[1::2] = 1
        game_features.append(Z)

    return game_features

def evaluation():
    train_color = randint(0, 1)
    if train_color == 0:
        model1 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_train.h5')
        model2 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')
    else:
        model1 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')
        model2 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_train.h5')

    # Initialize board and begin recording features for each board state
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    T = 0.00001

    # Play game
    while True:
        # Player 1 move
        print('Player 1 is thinking...')
        p1move, pi = mcts(board, model1, poss_moves, T=T)
        board.push(chess.Move.from_uci(p1move))

        if board.is_game_over():
            winner = 0
            break

        # Player 2 move
        print('Player 2 is thinking...')
        p2move, pi = mcts(board, model2, poss_moves, T=T)
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
