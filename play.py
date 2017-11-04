"""
This script allows users to play chess against algorithms
Current algorithm: Pure MCTS, no parallelization
"""
import chess
from mcts import mcts
import time

def main():
    print('Welcome to my chess game!', '\n')

    play_game = True
    while play_game:
        play()
        play_again = input('The game has ended. Would you like to play again? '
                          '[Y/n] ')
        if play_again.lower() is not 'y':
            play_game = False

    print('Thanks for playing!')


def player_color():
    player_color = input('Would you like to play as [W] or B? ')
    player_color = player_color.lower()
    if player_color == '' or player_color == 'w':
        print('You will play as White.', '\n')
        return 0
    elif player_color == 'b':
        print('You will play as Black.', '\n')
        return 1
    else:
        print('Invalid color selection.', '\n')
        return -1

def play():
    color = -1
    while color == -1:
        color = player_color()

    board = chess.Bitboard()
    print(board, '\n')

    # Handle player having first move
    if color == 0:
        move = input('Please enter your move in algebraic notation: ')
        legal = [board.san(move) for move in board.generate_legal_moves()]
        while move not in legal:
            print('\nMove illegal or not in algebraic notation.')
            move = input('Please enter a legal move in algebraic '
                             'notation: ')
        board.push_san(move)
        print(board)

    # Play game
    while not board.is_game_over():
        print('Brownie24 is thinking...')
        _, move = mcts(board)
        board.push(move)
        print(board)
        if board.is_game_over():
            break

        move = input('Please enter your move in algebraic notation: ')
        legal = [board.san(move) for move in board.generate_legal_moves()]
        while move not in legal:
            print('\nMove illegal or not in algebraic notation.')
            move = input('Please enter a legal move in algebraic '
                         'notation: ')
        board.push_san(move)
        print(board)

if __name__ == '__main__':
    main()