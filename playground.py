"""
Playground for testing things
"""

import pickle
import time
from self_play import self_play

def main():
    games = 0
    while games < 45:
        generate_game()
        time.sleep(15)

def generate_game():
    game_record = self_play()
    update(game_record)


def update(game_record):
    file_Name = "C:\Glenn\Stuff\Machine " \
                "Learning\chess\\records\\brownie24_self_play_records.pickle"
    fileObject = open(file_Name, 'rb')
    game_records = pickle.load(fileObject)
    fileObject.close()

    game_records[0] = game_records[0] + game_record[0]
    game_records[1] = game_records[1] + game_record[1]
    game_records[2] = game_records[2] + game_record[2]

    if len(game_records[0]) > 10000:
        del game_records[0][:200]
        del game_records[1][:200]
        del game_records[2][:200]

    fileObject = open(file_Name, 'wb')
    pickle.dump(game_records, fileObject)
    fileObject.close()


if __name__ == '__main__':
    main()