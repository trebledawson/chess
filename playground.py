"""
Playground for testing things
"""

import pickle
import time
from parallel_self_play import self_play

def main():
    record = [0, 0, 0]
    games = 0
    while games < 40:
        generate_game(record)
        time.sleep(15)
        games += 1
        print(record, '| Games:', games)


def generate_game(record):
    game_record, winner = self_play()
    if winner == -1:
        return
    else:
        update(game_record)
        if game_record[2][0] == 1:
            record[0] += 1
        elif game_record[2][0] == -1:
            record[1] += 1
        elif game_record[2][0] == -0.25:
            record[2] += 1


def update(game_record):
    file_Name = "C:\Glenn\Stuff\Machine " \
                "Learning\chess\\records\\brownie24_self_play_records.pickle"
    #if games != 0:
    fileObject = open(file_Name, 'rb')
    game_records = pickle.load(fileObject)
    fileObject.close()

    game_records[0] = game_records[0] + game_record[0]
    game_records[1] = game_records[1] + game_record[1]
    game_records[2] = game_records[2] + game_record[2]

    print(len(game_records[0]))
    if len(game_records[0]) > 50000:
        del game_records[0][:200]
        del game_records[1][:200]
        del game_records[2][:200]
    print(len(game_records[0]))
    fileObject = open(file_Name, 'wb')
    pickle.dump(game_records, fileObject)
    fileObject.close()
    '''
    # This only executes for the first entry in the records file. No longer 
    # necessary after at least one game has been recorded
    else:
        fileObject = open(file_Name, 'wb')
        pickle.dump(game_record, fileObject)
        fileObject.close()
    '''

if __name__ == '__main__':
    main()
