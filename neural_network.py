"""
Neural network architecture is similar to that used by Google DeepMind's
AlphaGo Zero.

Input features:
-8x8 plane of white piece locations
-8x8 plane of black piece locations
-1x1 vector containing the current player

The piece location planes are encoded as follows:
Empty spaces = 0
Pawns        = 1
Bishops      = 2
Knights      = 3
Rooks        = 4
Queens       = 5
King         = 6

The en passant plane is encoded as follows:
Empty spaces         = 0
En passant available = 1

The current player vector is encoded as follows:
White = 0
Black = 1

The castling vector is encoded as follows:
White kingside  = index 0
White queenside = index 1
Black kingside  = index 2
Black queenside = index 3
If castling is available, the appropriate index = 1, else 0.
"""

import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(5)
tf.set_random_seed(42)
rn.seed(3901)
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Add, Flatten, Concatenate, \
    Activation, BatchNormalization
from keras.regularizers import l2

def main():
    # Planar inputs containing white pieces and black pieces information
    w = Input((8, 8, 1))
    b = Input((8, 8, 1))

    # Layer 0: Concatenate feature plane inputs
    input = Concatenate(axis=3)([w, b])

    # Layer 1: Pass through convolutional layer
    a1 = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(input)
    a2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(a1)
    a3 = Activation('relu')(a2)

    # Layers 2-4: Residual blocks
    r1a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(a3)
    r1b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r1a)
    r1c = Activation('relu')(r1b)
    r1d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r1c)
    r1e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r1d)
    r1f = Add()([r1e, a3])
    r1g = Activation('relu')(r1f)

    r2a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r1g)
    r2b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r2a)
    r2c = Activation('relu')(r2b)
    r2d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r2c)
    r2e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r2d)
    r2f = Add()([r2e, r1g])
    r2g = Activation('relu')(r2f)

    r3a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r2g)
    r3b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r3a)
    r3c = Activation('relu')(r3b)
    r3d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r3c)
    r3e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r3d)
    r3f = Add()([r3e, r2g])
    r3g = Activation('relu')(r3f)

    # Scalar inputs containing current player information
    p = Input((1,))

    # Policy layers
    p1 = Conv2D(2, 1, activity_regularizer=l2(l=0.0001))(r3g)
    p2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(p1)
    p3 = Activation('relu')(p2)
    p4 = Flatten()(p3)
    p5 = Concatenate()([p, p4])
    p6 = Dense(256, activity_regularizer=l2(l=0.0001))(p5)
    p7 = Activation('relu')(p6)
    p8 = Dense(1968, activity_regularizer=l2(l=0.0001))(p7)
    probabilities = Activation('softmax')(p8)

    # Value layers
    v1 = Conv2D(1, 1, activity_regularizer=l2(l=0.0001))(r3g)
    v2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(v1)
    v3 = Activation('relu')(v2)
    v4 = Flatten()(v3)
    v5 = Concatenate()([p, v4])
    v6 = Dense(256, activity_regularizer=l2(l=0.0001))(v5)
    v7 = Activation('relu')(v6)
    v8 = Dense(1, activity_regularizer=l2(l=0.0001))(v7)
    value = Activation('tanh')(v8)

    # Compilation
    model = Model(inputs=[w, b, p], outputs=[probabilities, value])

    model.compile(optimizer='nadam', loss={'activation_10':
                                               'categorical_crossentropy',
                                       'activation_13':
                                               'mean_squared_error'})
    model.summary()

    model.save(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_train.h5')

    model.save(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_live.h5')

if __name__ == '__main__':
    main()
