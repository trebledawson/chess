import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(5)
tf.set_random_seed(42)
rn.seed(3901)
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, Add, Flatten, Concatenate, \
    Activation, BatchNormalization

"""
This network architecture is similar to that used by Google DeepMind's
AlphaGo Zero.
"""
# Layer 0: Concatenate feature plane inputs
w = Input((8, 8, 1))
b = Input((8, 8, 1))
ep = Input((8, 8, 1))
input = Concatenate(axis=3)([w, b, ep])

# Layer 1: Pass through convolutional layer
a1 = Conv2D(256, 3, padding='same')(input)
a2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(a1)
a3 = Activation('relu')(a2)

# Layers 2-4: Residual blocks
r1a = Conv2D(256, 3, padding='same')(a3)
r1b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r1a)
r1c = Activation('relu')(r1b)
r1d = Conv2D(256, 3, padding='same')(r1c)
r1e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r1d)
r1f = Add()([r1e, a3])
r1g = Activation('relu')(r1f)

r2a = Conv2D(256, 3, padding='same')(r1g)
r2b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r2a)
r2c = Activation('relu')(r2b)
r2d = Conv2D(256, 3, padding='same')(r2c)
r2e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r2d)
r2f = Add()([r2e, r1g])
r2g = Activation('relu')(r2f)

r3a = Conv2D(256, 3, padding='same')(r2g)
r3b = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r3a)
r3c = Activation('relu')(r3b)
r3d = Conv2D(256, 3, padding='same')(r3c)
r3e = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(r3d)
r3f = Add()([r3e, r2g])
r3g = Activation('relu')(r3f)

# Define vector inputs
p = Input((1,))
c = Input((4,))

# Policy layers
p1 = Conv2D(2, 1)(r3g)
p2 = BatchNormalization(axis=1, momentum=0.9, epsilon=0.00001)(p1)
p3 = Activation('relu')(p2)
p4 = Flatten()(p3)
p5 = Concatenate()([p, c, p4])
p6 = Dense(256, activation='relu')(p5)
out = Dense(1, activation = 'tanh')(p6)

# Compilation
model = Model(inputs=[w, b, p, c, ep], outputs=out)
model.compile(optimizer='nadam', loss='mean_squared_error')
model.summary()

model.save(filepath='G:\Glenn\Misc\Machine '
                    'Learning\Projects\chess\models\model.h5')