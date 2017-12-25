"""
Self-play arena for Brownie24.
"""
import numpy as np
import chess
import time
from random import randint
from parallel_mcts import mcts
#from numba_mcts import mcts
#from cchess_tools import mcts
from tools import all_possible_moves, SearchTree, features
from copy import deepcopy
from multiprocessing import Process, Pipe, cpu_count

# This self-play function is used to generate training data
def self_play():
    game_start = time.time()

    # Initialize neural network daemon
    modelpath = '.\models\model_live.h5'
    pipes_net = []
    pipes_sim = []
    for worker in range(cpu_count() - 1):
        p1, p2 = Pipe()
        pipes_net.append(p1)
        pipes_sim.append(p2)
    nn_p = Process(target=nn_daemon, args=(modelpath, pipes_net))
    nn_p.daemon = True
    nn_p.start()

    # Initialize board and begin recording features for each board state
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    game_record = [[board.fen()], [], []]
    p1tree = SearchTree()
    p2tree = SearchTree()
    move = 1
    T = 1

    # Start daemon
    feats = features(board.fen())
    feats = feats.reshape(1, 14, 8, 8)
    pipes_sim[0].send(feats)
    while not pipes_sim[0].poll():
        time.sleep(0.0000001)
    prior_, value_ = pipes_sim[0].recv()

    del prior_
    del value_

    # Play game and record board state features for each move
    print('Game start.')
    while True:
        # Determine temperature coefficient by game length
        if move > 10:
            T = 0.1

        # Player 1 move
        print('Player 1 is thinking...')
        p1move, pi, p1tree, index, Q = mcts(board, poss_moves, pipes_sim, T=T,
                                         tree=p1tree)
        board.push(chess.Move.from_uci(p1move))
        game_record[0].append(board.fen())
        game_record[1].append(deepcopy(pi))

        print(board)
        print('Player 1: ', move, '. ', p1move, ' | Q: ', Q, '\n', sep='')

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
        p2move, pi, p2tree, index, Q = mcts(board, poss_moves, pipes_sim, T=T,
                                         tree=p2tree)
        board.push(chess.Move.from_uci(p2move))
        game_record[0].append(board.fen())
        game_record[1].append(deepcopy(pi))

        print(board)
        print('Player 2: ', move, '... ', p2move, ' | Q: ', Q, '\n', sep='')

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

    # Kill neural network daemon
    nn_p.terminate()

    # Delete final board state from game record
    del game_record[0][-1]

    # Assign rewards and penalties
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

    return game_record, winner


# This self-play function is used to determine the new generator model
def evaluation():
    train_color = randint(0, 1)
    if train_color == 0:
        model1path = '.\models\model_train.h5'
        model2path = '.\models\model_live.h5'
        print('Evaluation network plays as White. Current generator network '
              'plays as Black.')
        player_1 = 'Evaluator'
        player_2 = 'Generator'
    else:
        model1path = '.\models\model_live.h5'
        model2path = '.\models\model_train.h5'
        print('Current generator network plays as White. Evaluation network '
              'plays as Black.')
        player_1 = 'Generator'
        player_2 = 'Evaluator'

    # Initialize neural network daemon for both players
    pipes_net1 = []
    pipes_sim1 = []
    pipes_net2 = []
    pipes_sim2 = []
    for worker in range(cpu_count() - 2):
        # Player 1 pipes
        p1, p2 = Pipe()
        pipes_net1.append(p1)
        pipes_sim1.append(p2)
        # Player 2 pipes
        p3, p4 = Pipe()
        pipes_net2.append(p3)
        pipes_sim2.append(p4)

    nn_p1 = Process(target=nn_daemon, args=(model1path, pipes_net1))
    nn_p2 = Process(target=nn_daemon, args=(model2path, pipes_net2))
    nn_p1.daemon = True
    nn_p2.daemon = True
    nn_p1.start()
    nn_p2.start()

    # Initialize board and game variables
    poss_moves = all_possible_moves()
    board = chess.Bitboard()
    p1tree = SearchTree()
    p2tree = SearchTree()
    move = 1
    T = 0.1  # Temperature coefficient is low for entire evaluation

    # Start daemon
    feats = features(board.fen())
    feats = feats.reshape(1, 14, 8, 8)
    pipes_sim1[0].send(feats)
    pipes_sim2[0].send(feats)
    while not pipes_sim1[0].poll():
        time.sleep(0.0000001)
    while not pipes_sim2[0].poll():
        time.sleep(0.0000001)
    prior_, value_ = pipes_sim1[0].recv()
    prior_, value_ = pipes_sim2[0].recv()

    del prior_
    del value_

    # Play game and record board state features for each move
    print('Game start.')
    while True:
        # Player 1 move
        print(player_1, 'is thinking...')
        p1move, pi, p1tree, index, Q = mcts(board, poss_moves, pipes_sim1, T=T,
                                         tree=p1tree)
        board.push(chess.Move.from_uci(p1move))
        print(board)
        print(player_1, ': ', move, '. ', p1move, ' | Q: ', Q, '\n', sep='')

        # Game ending conditions
        if board.is_game_over():
            if board.is_checkmate():
                winner = 0
                print('Winner:', player_1)
                break
            else:
                winner = -1
                print('Game drawn.')
                break

        if move != 1:
            p2tree = p2tree.nodes[index]

        # Player 2 move
        print(player_2, 'is thinking...')
        p2move, pi, p2tree, index, Q = mcts(board, poss_moves, pipes_sim2, T=T,
                                         tree=p2tree)
        board.push(chess.Move.from_uci(p2move))
        print(board)
        print(player_2, ': ', move, '... ', p2move, ' | Q: ', Q, '\n', sep='')

        if board.is_game_over():
            if board.is_checkmate():
                winner = 1
                print('Winner:', player_2)
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

    # Kill neural network daemons
    nn_p1.terminate()
    nn_p2.terminate()

    # Determine trained network's performance
    if winner == -1:
        return 0.5
    elif winner == train_color:
        return 1
    else:
        return 0


def nn_daemon(modelpath, pipes_net):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as ktf
    from keras.models import load_model

    def get_session(gpu_fraction=0.45):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())

    model = load_model(filepath=modelpath)

    while True:
        for pipe in pipes_net:
            if not pipe.poll():
                continue
            state_features = pipe.recv()
            priors, value = model.predict(state_features, verbose=0)
            priors = np.ravel(priors)
            value = value[0][0]
            pipe.send((priors, value))
