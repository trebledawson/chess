###############################################################################
#  Brownie24 - A Chess-Playing AI Using Neural Networks and MCTS              #
#  Author: Glenn Dawson                                                       #
#  MCTS and Neural Network architecture adapted from DeepMind's AlphaGo Zero  #
###############################################################################
import numpy as np
from self_play import self_play, evaluation
from random import randint
from keras.models import load_model

# TODO: Implement training using parallelization

def main():
    game_records = []

    # Next three loops should get parallelized:
    # -----------------------------------------

    # Self-play
    game_count = 0
    while game_count < 2500:
        game_record = self_play()
        game_records.append(game_record)
        if len(game_records) > 10000:
            del game_records[randint(0, 5000)]

        game_count += 1

    # Optimization
    training = True
    while training == True:
        model = load_model(filepath=
                           'G:\Glenn\Misc\Machine '
                           'Learning\Projects\chess\models\model_train.h5')
        data = np.array(game_records[:, :-2])
        pis = np.array(game_records[:, -2])
        results = np.array(game_records[:, -1])

        indices = np.random.randint(0, len(pis), size=(1, 2048))
        data = data[indices]
        pis = pis[indices]
        results = results[indices]

        model.fit(data, [pis, results], verbose=0)
        model.save(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_train.h5')

    # Evaluate
    evaluation_count = 0
    train_wins = 0
    while evaluation_count < 200:
        train_win = evaluation()
        train_wins += train_win

    if train_wins >= 110:
        model.save(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')



if __name__ == '__main__':
    main()