###############################################################################
#  Brownie24 - A Chess-Playing AI Using Neural Networks and MCTS              #
#  Author: Glenn Dawson                                                       #
###############################################################################
"""
TODO:
- Implement neural network
- Interface NN with MCTS
"""


import chess
from mcts import mcts
from tools import features

def main():
    """
    Pass legal moves into MCTS, iterate. Select move according to a_t ~ pi_t
    
    MCTS:
    Returns pi, z
    pi is 
    """

    """
    Train convolutional neural network as in AlphaGo Zero:
    -Take board position (FEN) features as input
    -Return probabilities p_ and win probability v
    -Update NN parameters to match p_ to pi_t, minimize error between z and 
     v

    See: Fig. 1 and 2 in AlphaGo Zero paper
    
    p_ is a vector containing probabilities of each legal move
    v is a scalar estimating win probability
    """



if __name__ == '__main__':
    main()