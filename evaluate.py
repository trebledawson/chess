from parallel_self_play import evaluation
import time

def main():
    from keras.models import load_model
    train_wins = 0.
    record = [0, 0, 0]
    for evaluation_game in range(100):
        train_win = evaluation()
        train_wins += train_win
        if train_win == 1:
            record[0] += 1
        elif train_win == 0:
            record[1] += 1
        else:
            record[2] += 1
        print('Training score:', train_wins, '| Games:', evaluation_game + 1,
              '| Record [W, L, D]:', record)

        time.sleep(15)

    print(train_wins)

    if train_wins >= 55:
        model = load_model('.\models\model_train.h5')
        model.save(filepath='.\models\model_live.h5')
        del model

if __name__ == '__main__':
    main()
