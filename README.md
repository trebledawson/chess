# Chess AI modeled after DeepMind's AlphaGo Zero.

# TODO:
* Train neural network.

# Dependencies:
* Python 3.6
* python-chess 0.4.1
* Keras 
* TensorFlow-GPU
* CUDA 7 or higher
* cuDNN 5 or higher

# Using cleaning.py and GM_training.py
* Required: GMbothall.pgn (found here: https://chess-db.com/public/downloads/gamesfordownload.jsp)
* Note: This will generate a large number of large text files, on the order of 200 GB. 

# Self-Play
* playground.py will generate games by having the best current model (model\_live.h5) play against itself
* evaluate.py will generate games by having the best current model (model\_live.h5) play against the trained model (model\_train.h5)
