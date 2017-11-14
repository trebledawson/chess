"""
Self-play arena for Brownie24.
"""
import numpy as np
import chess
import time
from random import randint
from keras.models import load_model
from linked_mcts import mcts
#from parallelization_test import mcts
from tools import all_possible_moves, SearchTree
from copy import deepcopy

# This self-play function is used to generate training data
def self_play():
    game_start = time.time()
    model1 = load_model(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_live.h5')
    #model1path = 'C:\Glenn\Stuff\Machine Learning\chess\models\model_live.h5'
    model2 = load_model(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_live.h5')
    #model2path = 'C:\Glenn\Stuff\Machine Learning\chess\models\model_live.h5'

    # Initialize board and begin recording features for each board state
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    game_record = [[board.fen()], [], []]
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
        p1move, pi, p1tree, index = mcts(board, model1, poss_moves,
                                         T=T, tree=p1tree)
        board.push(chess.Move.from_uci(p1move))
        game_record[0].append(board.fen())
        game_record[1].append(deepcopy(pi))

        print(board)
        print('Player 1: ', move, '. ', p1move, '\n', sep='')

        # Game ending conditions
        if board.is_game_over():
            if board.is_checkmate():
                winner = 0
                print('Winner: White.')
                print('Game duration:', time.time() - game_start, 'seconds. \n')
                break
            else:
                winner = -1
                print('Game drawn.')
                print('Game duration:', time.time() - game_start, 'seconds. \n')
                break

        if move != 1:
            # Update Player 2's decision tree with Player 1's move
            p2tree = p2tree.nodes[index]

        # Player 2 move
        print('Player 2 is thinking...')
        p2move, pi, p2tree, index = mcts(board, model2, poss_moves,
                                         T=T, tree=p2tree)
        board.push(chess.Move.from_uci(p2move))
        game_record[0].append(board.fen())
        game_record[1].append(deepcopy(pi))

        print(board)
        print('Player 2: ', move, '... ', p2move, '\n', sep='')

        if board.is_game_over():
            if board.is_checkmate():
                winner = 1
                print('Winner: Black.')
                print('Game duration:', time.time() - game_start, 'seconds.\n')
                break
            else:
                winner = -1
                print('Game drawn.')
                print('Game duration:', time.time() - game_start, 'seconds.\n')
                break

        # Check if game is over by length
        if move == 100:
            winner = -1
            print('Game drawn by length.')
            print('Game duration:', time.time() - game_start, 'seconds. \n')
            break

        # Update Player 1 decision tree with Player 2's move
        p1tree = p1tree.nodes[index]

        move += 1

    del model1
    del model2
    del game_record[0][-1]

    if winner == 0:
        # Reward Player 1, penalize Player 2
        Z = np.zeros(len(game_record[0]))
        Z[::2] = 1
        Z[1::2] = -1
        for z in Z.tolist():
            game_record[2].append(z)
    elif winner == 1:
        # Penalize Player 1, penalize Player 2
        Z = np.zeros(len(game_record[0]))
        Z[::2] = -1
        Z[1::2] = 1
        for z in Z.tolist():
            game_record[2].append(z)
    else:
        # Slightly penalize draws
        Z = np.full(len(game_record[0]), -0.25)
        for z in Z.tolist():
            game_record[2].append(z)

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

    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    p1tree = SearchTree()
    p2tree = SearchTree()
    move = 1
    T = 0.1  # Temperature coefficient is low for entire evaluation

    # Play game and record board state features for each move
    print('Game start.')
    while True:
        # Player 1 move
        print('Player 1 is thinking...')
        p1move, pi, p1tree, index = mcts(board, model1, poss_moves, T=T,
                                         tree=p1tree)
        board.push(chess.Move.from_uci(p1move))
        print(board)
        print('Player 1: ', move, '. ', p1move, '\n', sep='')

        # Game ending conditions
        if board.is_game_over():
            if board.is_checkmate():
                winner = 0
                print('Winner: White.')
                break
            else:
                winner = -1
                print('Game drawn.')
                break

        if move != 1:
            # Update Player 2's decision tree with Player 1's move
            p2tree = p2tree.nodes[index]

        # Player 2 move
        print('Player 2 is thinking...')
        p2move, pi, p2tree, index = mcts(board, model2, poss_moves, T=T,
                                         tree=p2tree)
        board.push(chess.Move.from_uci(p2move))
        print(board)
        print('Player 2: ', move, '... ', p2move, '\n', sep='')

        if board.is_game_over():
            if board.is_checkmate():
                winner = 1
                print('Winner: Black.')
                break
            else:
                winner = -1
                print('Game drawn.')
                break

        # Check if game is over by length
        if move == 100:
            winner = -1
            print('Game drawn by length.')
            break

        # Update Player 1 decision tree with Player 2's move
        p1tree = p1tree.nodes[index]

        move += 1

    del model1
    del model2

    if winner == -1:
        return 0.5
    elif winner == train_color:
        return 1
    else:
        return 0
