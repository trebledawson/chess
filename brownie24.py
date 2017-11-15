###############################################################################
#  Brownie24 - A Chess-Playing AI Using Neural Networks and MCTS              #
#  Author: Glenn Dawson                                                       #
#  MCTS and Neural Network architecture adapted from DeepMind's AlphaGo Zero  #
###############################################################################
import numpy as np
from parallel_self_play import self_play, evaluation
from random import randint
from keras.models import load_model
import pickle

# TODO: Implement training using parallelization

def main():
    """
    TODO:
    [X] Write script that generates games of self-play data
        See: playground.py
    [X] Write script that trains new neural network using generated self-play data
        See: training.py
    [X] Write script that evaluates trained network against current generator network
        See: playground.py
    [ ] Parallelize these scripts to run in tandem
    """

    


if __name__ == '__main__':
    main()