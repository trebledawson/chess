import chess
import numpy as np
from tools import all_possible_moves
import pickle

fp = open('C:\Glenn\Stuff\Machine Learning\chess\Data\GMallboth.pgn')

game_records = [[], [], []]
game_length = 0
poss = all_possible_moves()
for line, data in enumerate(fp):
    if (line - 11) % 16 == 0:
        game_length = 0
        board = chess.Bitboard()
        game_records[0] = game_records[0] + [board.fen()]
        print(line + 1, ':', data)
        moves = data.split()
        del moves[::3]
        for move in moves:
            board.push_san(move)
            game_records[0] = game_records[0] + [board.fen()]

            move_ = board.peek().uci()
            index = poss.index(move_)
            pi = np.zeros(1968)
            pi[index] = 1

            game_records[1] = game_records[1] + [pi]

            game_length += 1
        del game_records[0][-1]


    elif (line - 13) % 16 == 0:
        print(line + 1, ':', data)
        data = data.split()
        if data[0] == '1-0':
            winner = 0
        elif data[0] == '0-1':
            winner = 1
        else:
            winner = -1

        # Assign rewards and penalties
        if winner == 0:
            # Reward Player 1, penalize Player 2
            Z = np.zeros(game_length)
            Z[::2] = 1
            Z[1::2] = -1
            Z = Z.tolist()
            game_records[2] = game_records[2] + Z
        elif winner == 1:
            # Penalize Player 1, penalize Player 2
            Z = np.zeros(game_length)
            Z[::2] = -1
            Z[1::2] = 1
            Z = Z.tolist()
            game_records[2] = game_records[2] + Z
        else:
            # Slightly penalize draws
            Z = np.full(game_length, -0.25)
            Z = Z.tolist()
            game_records[2] = game_records[2] + Z

        print('Positions:', len(game_records[0]))
        print('Probabilities:', len(game_records[1]))
        print('Results:', len(game_records[2]))

fp.close()

fp = open('C:\Glenn\Stuff\Machine Learning\chess\Data\GMdata.pickle', 'wb')
pickle.dump(game_records, fp)
fp.close()

print('Done.')
