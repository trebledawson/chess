import pickle
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import KFold
from tools import features
from random import shuffle
import gc
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def main():
    for i in range(100):
        print('Iteration:', i)
        train()

def train():
    # Iterate over all GMdata files

    training_set = 1
    files = list(range(1, 287))
    shuffle(files)
    for file_num in files:
        print('Training set:', training_set)
        kf = KFold(n_splits=15, shuffle=True)
        early = EarlyStopping(monitor='val_activation_37_loss', min_delta=0.001,
                              patience=10, verbose=1)
        tensorboard = TensorBoard(log_dir='.\TensorBoard')

        filename = '.\Data\GMdata' + str(file_num) + '.pickle'
        file_object = open(filename, 'rb')
        game_record = pickle.load(file_object)
        file_object.close()
        del filename
        del file_object

        for ignore_, train_idx in kf.split(game_record[0]):
            del ignore_
            feats = np.zeros((len(train_idx), 14, 8, 8))
            pis = np.zeros((len(train_idx), 1968))
            results = np.zeros((len(train_idx), 1))

            count = 0
            for index in train_idx:
                feat = features(game_record[0][index])
                feats[count] = feat
                pis[count] = game_record[1][index]
                results[count] = game_record[2][index]

                count += 1

            ktf.set_session(get_session())
            model = load_model(filepath='.\models\model_train.h5')
            model.fit(feats, [pis, results], batch_size=200,
                      epochs=300, verbose=2, callbacks=[early, tensorboard],
                      validation_split=0.3)
            model.save(filepath='.\models\model_train.h5')

            del model

            break

        del game_record
        print('Training epoch:', training_set, 'completed. Training set:', file_num, '\n')
        training_set += 1
        gc.collect()
        time.sleep(15)

    print('Training complete.')

    '''
    # K-fold cross validation on random subset of training data
    subset = np.random.randint(1, high=181, size=1)
    filename = '.\Data\GMdata' + str(subset) + '.pickle'
    file_object = open(filename, 'rb')
    game_record = pickle.load(file_object)
    file_object.close()

    kf = KFold(n_splits=65, shuffle=True)
    model = load_model(filepath='.\models\model_train.h5')
    scores = []
    for ignore_, train_idx in kf.split(game_record):
        del ignore_
        count = 0
        for index in train_idx:
            w, b, p = features(game_record[0][index])
            ws[count] = w
            bs[count] = b
            ps[count] = p
            pis[count] = game_record[1][index]
            results[count] = game_record[2][index]

            count += 1
        score = model.evaluate([ws, bs, ps], [pis, results], verbose=0)
        scores.append(score)

    score = 1 - (sum(scores) / len(scores))
    print('Grandmaster accuracy', score * 100, 'percent.')
    '''


def get_session(gpu_fraction=0.75):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == '__main__':
    main()
