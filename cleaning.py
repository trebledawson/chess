import chess
import numpy as np
from tools import all_possible_moves
import pickle
import time

fp = open('.\Data\GMallboth.pgn')

game_records = [[], [], []]
game_length = 0
white = 0
black = 0
poss = all_possible_moves()
games = 0
dump_check = 0
first_check = True
file_number = 1
illegal_check = False
start = time.time()
for line, data in enumerate(fp):
    if (line - 11) % 16 == 0:               # Lines containing game records
        game_length = 0
        states = []
        pis = []
        board = chess.Bitboard()
        game_records[0] = game_records[0] + [board.fen()]
        print(line + 1, ':', data)
        moves = data.split()
        del moves[::3]
        for move in moves:
            try:
                # If move is legal, play move, extract FEN, and set pi for
                # that position to 0.9 for the played move. For all other
                # legal moves, set the probabilities to 1/(n-1) where n is
                # the number of legal moves.
                # If there is only one possible move, set that probability to 1
                indices = [poss.index(move_.uci()) for move_ in
                           board.generate_legal_moves()]

                board.push_san(move)
                states.append(board.fen())

                move_ = board.peek().uci()
                index = poss.index(move_)
                pi = np.zeros(1968)
                if len(indices) > 1:
                    prob = 0.1 / len(indices)
                    for idx in indices:
                        pi[idx] = prob
                    pi[index] = 0.9
                else:
                    pi[index] = 1

                pis.append(pi)

                game_length += 1
            except:
                # If move is illegal, delete current game record and skip
                # results processing
                print('Illegal move detected at move', game_length / 2)
                illegal_check = True
                break

    elif (line - 13) % 16 == 0:             # Lines containing results of games analyzed in previous step
        if illegal_check:
            # Check if illegal move encountered
            game_records = [[], [], []]
            illegal_check = False
            continue

        print(line + 1, ':', data)
        data = data.split()
        if data[0] == '1-0':
            winner = 0
        elif data[0] == '0-1':
            winner = 1
        else:
            winner = -1

        # Assign rewards and penalties
        del game_records[0][-1]
        if winner == 0:
            # Write board states and pis to master record
            game_records[0] = game_records[0] + states
            game_records[1] = game_records[1] + pis

            # Reward Player 1, penalize Player 2
            Z = np.zeros(game_length)
            Z[::2] = 1
            Z[1::2] = -1
            game_records[2] = game_records[2] + Z.tolist()

            # Internal
            white += 1
        elif winner == 1:
            # Write board states and pis to master record
            game_records[0] = game_records[0] + states
            game_records[1] = game_records[1] + pis

            # Penalize Player 1, penalize Player 2
            Z = np.zeros(game_length)
            Z[::2] = -1
            Z[1::2] = 1
            game_records[2] = game_records[2] + Z.tolist()

            # Internal
            black += 1
        else:
            # Throw out draws
            continue

        games += 1
        print('Games:', games, '\n')

        dump_check += 1

        filename = '.\Data\GMdata' + str(file_number) + \
                   '.pickle'

        if first_check:
            # If this is the first game in a new set, open a new file
            picklefile = open(filename, 'wb')
            pickle.dump(game_records, picklefile)
            picklefile.close()
            first_check = False

        if dump_check == 400:
            # Each 400 games, write to pickle and delete active memory
            picklefile = open(filename, 'rb')
            records = pickle.load(picklefile)
            picklefile.close()

            records[0] = records[0] + game_records[0]
            records[1] = records[1] + game_records[1]
            records[2] = records[2] + game_records[2]

            print('Positions:', len(records[0]))
            print('Probabilities:', len(records[1]))
            print('Results:', len(records[2]))
            print('White wins:', white, '| Black wins:', black)
            print('Elapsed time: ', time.time() - start, 'seconds.\n')

            picklefile = open(filename, 'wb')
            pickle.dump(records, picklefile)
            picklefile.close()

            game_records = [[], [], []]

            dump_check = 0

            if len(records[0]) > 20000:
                # If pickle files exceed 20000 positions, start a new file
                first_check = True
                file_number += 1
fp.close()

# Write final games to pickle
filename = '.\Data\GMdata' + str(file_number) + '.pickle'
picklefile = open(filename, 'rb')
records = pickle.load(picklefile)
picklefile.close()

records[0] = records[0] + game_records[0]
records[1] = records[1] + game_records[1]
records[2] = records[2] + game_records[2]

print('Positions:', len(records[0]))

picklefile = open(filename, 'wb')
pickle.dump(records, picklefile)
picklefile.close()

print('Elapsed time: ', time.time() - start, 'seconds.\n')


print('Done.')
