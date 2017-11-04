"""
Self-play arena for Brownie24.
-Loads current network model
-Plays a game to completion
-Updates network model
"""

from keras.models import load_model
from linked_mcts import mcts
import chess
from tools import features
import time

while True:
    game_start = time.time()
    model1 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model.h5')
    model2 = load_model(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model.h5')

    # Initialize board and begin recording features for each board state
    board = chess.Bitboard()
    w0, b0, p0, c0, ep0 = features(board)
    ws = [w0]
    bs = [b0]
    ps = [p0]
    cs = [c0]
    eps = [ep0]

    # Play game and record board state features for each move
    while True:
        # Player 1 move
        p1move = mcts(board, model1)
        board.push(p1move)
        w, b, p, c, ep = features(board)
        ws.append(w)
        bs.append(b)
        ps.append(p)
        cs.append(c)
        eps.append(ep)

        if board.is_game_over():
            winner = 1
            break

        # Player 2 move
        p2move = mcts(board, model2)
        board.push(p2move)
        w, b, p, c, ep = features(board)
        ws.append(w)
        bs.append(b)
        ps.append(p)
        cs.append(c)
        eps.append(ep)

        if board.is_game_over():
            winner = 2
            break

    ws = ws.reshape(len(ws), 8, 8, 1)
    bs = bs.reshape(len(bs), 8, 8, 1)
    ps = ps.reshape(len(ps), 1)
    cs = cs.reshape(len(cs), 4)
    eps = eps.reshape(len(eps), 8, 8, 1)
    train = [ws, bs, ps, cs, eps]

    if winner == 1:
        # For each state in the game, update neural network
        model2 = model2.fit(train, 0, verbose=0)
        model2.save(filepath='G:\Glenn\Misc\Machine '
                    'Learning\Projects\chess\models\model.h5')
        del model1
        del model2

    else:
        # For each state in the game, update neural network
        model1 = model1.fit(train, 0, verbose=0)
        model1.save(filepath='G:\Glenn\Misc\Machine '
                             'Learning\Projects\chess\models\model.h5')
        del model1
        del model2

    print('Game duration:', time.time() - game_start, 'seconds.')
