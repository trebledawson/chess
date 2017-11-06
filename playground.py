"""
Playground for testing things
"""
import chess
from self_play import self_play

def main():
    game_records = []
    game_record = self_play()
    game_records.append(game_record)

if __name__ == '__main__':
    main()