"""
Neural network architecture is similar to that used by Google DeepMind's
AlphaGo Zero.

Binary feature input planes:
1.) 8x8 plane of white pawn locations
2.) 8x8 plane of white knight locations
3.) 8x8 plane of white bishop locations
4.) 8x8 plane of white rook locations
5.) 8x8 plane of white queen locations
6.) 8x8 plane of white king location
7.) 8x8 plane of black pawn locations
8.) 8x8 plane of black knight locations
9.) 8x8 plane of black bishop locations
10.) 8x8 plane of black rook locations
11.) 8x8 plane of black queen locations
12.) 8x8 plane of black king location
13.) 8x8 plane of empty square locations
14.) 8x8 plane of current player information (0 for White, 1 for Black)
"""

import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(5)
tf.set_random_seed(42)
rn.seed(3901)
from keras.models import Model
from keras.backend import set_image_data_format
from keras.layers import Conv2D, Dense, Input, Add, Flatten, Activation, \
    BatchNormalization, Dropout
from keras.regularizers import l2

def main():
    set_image_data_format('channels_first')

    # Planar inputs containing white pieces and black pieces information
    features = Input((14, 8, 8))

    # Layer 1: Pass through convolutional layer
    a1 = Conv2D(256, 3, padding='same',
                activity_regularizer=l2(l=0.0001))(features)
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
    r3f = Add()([r3e, r2g, r1g])
    r3g = Activation('relu')(r3f)

    r4a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r3g)
    r4b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r4a)
    r4c = Activation('relu')(r4b)
    r4d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r4c)
    r4e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r4d)
    r4f = Add()([r4e, r3g, r2g, r1g])
    r4g = Activation('relu')(r4f)

    r5a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r4g)
    r5b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r5a)
    r5c = Activation('relu')(r5b)
    r5d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r5c)
    r5e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r5d)
    r5f = Add()([r5e, r4g, r3g, r2g, r1g])
    r5g = Activation('relu')(r5f)

    r6a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r5g)
    r6b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r6a)
    r6c = Activation('relu')(r6b)
    r6d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r6c)
    r6e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r6d)
    r6f = Add()([r6e, r5g, r4g, r3g, r2g, r1g])
    r6g = Activation('relu')(r6f)

    r7a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r6g)
    r7b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r7a)
    r7c = Activation('relu')(r7b)
    r7d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r7c)
    r7e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r7d)
    r7f = Add()([r7e, r6g, r5g, r4g, r3g, r2g, r1g])
    r7g = Activation('relu')(r7f)

    r8a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r7g)
    r8b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r8a)
    r8c = Activation('relu')(r8b)
    r8d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r8c)
    r8e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r8d)
    r8f = Add()([r8e, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r8g = Activation('relu')(r8f)

    r9a = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r8g)
    r9b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r9a)
    r9c = Activation('relu')(r9b)
    r9d = Conv2D(256, 3, padding='same', activity_regularizer=l2(l=0.0001))(r9c)
    r9e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r9d)
    r9f = Add()([r9e, r8g, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r9g = Activation('relu')(r9f)

    r10a = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r9g)
    r10b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r10a)
    r10c = Activation('relu')(r10b)
    r10d = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r10c)
    r10e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r10d)
    r10f = Add()([r10e, r9g, r8g, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r10g = Activation('relu')(r10f)

    r11a = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r10g)
    r11b = BatchNormalization(axis=1, momentum=0.10, epsilon=0.00001)(r11a)
    r11c = Activation('relu')(r11b)
    r11d = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r11c)
    r11e = BatchNormalization(axis=1, momentum=0.10, epsilon=0.00001)(r11d)
    r11f = Add()([r11e, r10g, r9g, r8g, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r11g = Activation('relu')(r11f)

    r12a = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r11g)
    r12b = BatchNormalization(axis=1, momentum=0.11, epsilon=0.00001)(r12a)
    r12c = Activation('relu')(r12b)
    r12d = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r12c)
    r12e = BatchNormalization(axis=1, momentum=0.11, epsilon=0.00001)(r12d)
    r12f = Add()([r12e, r11g, r10g, r9g, r8g, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r12g = Activation('relu')(r12f)

    r13a = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r12g)
    r13b = BatchNormalization(axis=1, momentum=0.12, epsilon=0.00001)(r13a)
    r13c = Activation('relu')(r13b)
    r13d = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r13c)
    r13e = BatchNormalization(axis=1, momentum=0.12, epsilon=0.00001)(r13d)
    r13f = Add()([r13e, r12g, r11g, r10g, r9g, r8g, r7g, r6g, r5g, r4g, r3g,
                  r2g, r1g])
    r13g = Activation('relu')(r13f)

    r14a = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r13g)
    r14b = BatchNormalization(axis=1, momentum=0.13, epsilon=0.00001)(r14a)
    r14c = Activation('relu')(r14b)
    r14d = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r14c)
    r14e = BatchNormalization(axis=1, momentum=0.13, epsilon=0.00001)(r14d)
    r14f = Add()([r14e, r13g, r12g, r11g,
                  r10g, r9g, r8g, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r14g = Activation('relu')(r14f)

    r15a = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r14g)
    r15b = BatchNormalization(axis=1, momentum=0.14, epsilon=0.00001)(r15a)
    r15c = Activation('relu')(r15b)
    r15d = Conv2D(256, 3, padding='same',
                  activity_regularizer=l2(l=0.0001))(r15c)
    r15e = BatchNormalization(axis=1, momentum=0.14, epsilon=0.00001)(r15d)
    r15f = Add()([r15e, r14g, r13g, r12g, r11g,
                  r10g, r9g, r8g, r7g, r6g, r5g, r4g, r3g, r2g, r1g])
    r15g = Activation('relu')(r15f)

    # Policy layers
    p1 = Conv2D(64, 1, activity_regularizer=l2(l=0.0001))(r15g)
    p2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(p1)
    p3 = Activation('relu')(p2)
    p4 = Flatten()(p3)
    probabilities = Dense(1968, activity_regularizer=l2(l=0.0001))(p4)
    #probabilities = Activation('softmax')(p5)

    # Value layers
    v1 = Conv2D(64, 1, activity_regularizer=l2(l=0.0001))(r10g)
    v2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(v1)
    v3 = Activation('relu')(v2)
    v4 = Flatten()(v3)
    v5 = Dense(1024, activity_regularizer=l2(l=0.0001))(v4)
    v6 = Dropout(0.5)(v5)
    v7 = Activation('relu')(v6)
    v8 = Dense(256, activity_regularizer=l2(l=0.0001))(v7)
    v9 = Dropout(0.5)(v8)
    v10 = Activation('relu')(v9)
    v11 = Dense(64, activity_regularizer=l2(l=0.0001))(v10)
    v12 = Dropout(0.5)(v11)
    v13 = Activation('relu')(v12)
    v14 = Dense(1, activity_regularizer=l2(l=0.0001))(v13)
    value = Activation('tanh')(v14)

    # Compilation
    model = Model(inputs=features, outputs=[probabilities, value])

    model.compile(optimizer='nadam',
                  loss={'dense_1': 'categorical_crossentropy',
                        'activation_37': 'mean_squared_error'},
                  loss_weights={'dense_1': 0.05, 'activation_37': 0.95},
                  metrics={'dense_1': 'accuracy', 'activation_37': 'accuracy'})
    model.summary()

    model.save(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_train.h5')

    model.save(filepath='C:\Glenn\Stuff\Machine '
                        'Learning\chess\models\model_live.h5')

if __name__ == '__main__':
    main()
