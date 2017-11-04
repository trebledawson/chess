import numpy as np


def features(board):
    fen = board.fen().split()

    # Self-contained type checker
    def int_if_int(s):
        try:
            int(s)
            return int(s)
        except ValueError:
            return s

    # Convert board to array
    white = np.zeros(64).astype('float32')
    black = np.zeros(64).astype('float32')
    count = 0
    for letter in fen[0]:
        letter = int_if_int(letter)
        if letter in range(9):
            while letter > 0:
                white[count] = 0
                black[count] = 0
                letter -= 1
                count += 1
        elif letter == 'p':
            white[count] = 0
            black[count] = 1
            count += 1
        elif letter == 'P':
            white[count] = 1
            black[count] = 0
            count += 1
        elif letter == '/':
            continue
        elif letter == 'b':
            white[count] = 0
            black[count] = 2
            count += 1
        elif letter == 'B':
            white[count] = 2
            black[count] = 0
            count += 1
        elif letter == 'n':
            white[count] = 0
            black[count] = 3
            count += 1
        elif letter == 'N':
            white[count] = 3
            black[count] = 0
            count += 1
        elif letter == 'r':
            white[count] = 0
            black[count] = 4
            count += 1
        elif letter == 'R':
            white[count] = 4
            black[count] = 0
            count += 1
        elif letter == 'q':
            white[count] = 0
            black[count] = 5
            count += 1
        elif letter == 'Q':
            white[count] = 5
            black[count] = 0
            count += 1
        elif letter == 'k':
            white[count] = 0
            black[count] = 6
            count += 1
        elif letter == 'K':
            white[count] = 6
            black[count] = 0
            count += 1
    white = white.reshape(8, 8, 1)
    black = black.reshape(8, 8, 1)

    # Determine current player
    player = np.array(0)
    if fen[1] == 'b':
        player += 1

    # Determine castling
    castling = np.zeros(4).astype('float32')
    if 'K' in fen[2]:
        castling[0] = 1
    if 'Q' in fen[2]:
        castling[1] = 1
    if 'k' in fen[2]:
        castling[2] = 1
    if 'q' in fen[2]:
        castling[3] = 1

    # Determine en passant
    enpassant = np.zeros((8, 8)).astype('float32')
    if fen[3] is not '-':
        x = 0
        y = 0
        for letter in fen[3]:
            letter = int_if_int(letter)
            if letter in range(8):
                x = letter
            else:
                y = ord(letter) - 97
        enpassant[-x][y] = 1
    enpassant = enpassant.reshape(8, 8, 1)

    return white, black, player, castling, enpassant
