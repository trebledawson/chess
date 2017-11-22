"""
Training script. Imports records of self play, randomly samples positions,
their corresponding MCTS probabilities, and the final result of the game that
the position is drawn from.

The neural network is training to maximize the similarity of its
probabilities to the MCTS probabilities, and to minimize the difference
between the win probability "value" and the true winner of the game.

This training is done over 100 different sample batches. After 100 batches, the
trained network plays a series of games against the current generator network.
If the trained network wins at least 55% of these games, the trained network
becomes the new generator network. This process then repeats.
"""
import pickle
import numpy as np
from tools import features
from parallel_self_play import evaluation


def main():
    from keras.models import load_model
    from keras.callbacks import EarlyStopping

    file_Name = "C:\Glenn\Stuff\Machine " \
                "Learning\chess\\records\\brownie24_self_play_records.pickle"
    fileObject = open(file_Name, 'rb')
    game_records = pickle.load(fileObject)
    fileObject.close()

    early = EarlyStopping(patience=20, verbose=0)
    sample_size = 3000
    for training_epoch in range(100):
        print('Training epoch:', training_epoch, '\n')

        ws = np.zeros((sample_size, 8, 8, 1))
        bs = np.zeros((sample_size, 8, 8, 1))
        ps = np.zeros((sample_size, 1))
        pis = np.zeros((sample_size, 1968))
        results = np.zeros((sample_size, 1))
        sample_indices = np.random.randint(0, high=len(game_records[0]),
                                           size=sample_size)
        sample = 0
        while sample < sample_size:
            w, b, p = features(game_records[0][sample_indices[sample]])
            ws[sample] = w
            bs[sample] = b
            ps[sample] = p
            pis[sample] = game_records[1][sample_indices[sample]]
            results[sample] = game_records[2][sample_indices[sample]]
            sample += 1

        training_model = load_model(filepath='C:\Glenn\Stuff\Machine '
                                             'Learning\chess\models\model_train.h5')
        training_model.fit([ws, bs, ps], [pis, results], batch_size=200,
                           epochs=300, verbose=2, callbacks=[early],
                           validation_split=0.3)
        training_model.save(filepath='C:\Glenn\Stuff\Machine '
                            'Learning\chess\models\model_train.h5')
        del training_model

def evaluate():
    train_wins = 0.
    for evaluation_game in range(100):
        train_win = evaluation()
        train_wins += train_win

    print(train_wins)

    if train_wins >= 50:
        from keras.models import load_model
        model = load_model('G:\Glenn\Misc\Machine '
                           'Learning\Projects\chess\models\model_train.h5')
        model.save(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')
        del model


if __name__ == '__main__':
    main()
