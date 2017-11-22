from parallel_self_play import evaluation

def main():
    from keras.models import load_model
    train_wins = 0.
    for evaluation_game in range(100):
        train_win = evaluation()
        train_wins += train_win
        print('Training score:', train_wins, '| Games:', evaluation_game + 1)
        

    print(train_wins)

    if train_wins >= 55:
        model = load_model('G:\Glenn\Misc\Machine '
                           'Learning\Projects\chess\models\model_train.h5')
        model.save(filepath='G:\Glenn\Misc\Machine '
                            'Learning\Projects\chess\models\model_live.h5')
        del model

if __name__ == '__main__':
    main()
