import tensorflow  as tf 
from tensorflow.keras import layers, Sequential # type: ignore
from tensorflow.keras.layers import LeakyReLU, ReLU, BatchNormalization # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def ConvBlock(hidden, kernel, strides):
    convblock = Sequential()

    convblock.add(layers.Conv1DTranspose(hidden, kernel, strides, padding='same'))
    convblock.add(layers.ReLU())
    convblock.add(layers.BatchNormalization())

    return convblock

def ConvGenerator(noise_dim, feature_dim):
    model = Sequential()

    model.add(layers.Input((noise_dim,)))
    model.add(layers.Dense(feature_dim * 6))
    model.add(layers.Reshape((feature_dim // 6, 6 * 6)))

    model.add(ConvBlock(125, 2, 1))
    model.add(ConvBlock(75, 5, 2))
    model.add(ConvBlock(25, 7, 3))

    model.add(layers.Conv1D(1, 3, activation='tanh', padding='same'))
    model.add(layers.Flatten())

    return model

def NormalGenerator(noise_dim, feature_dim):

    model = Sequential()

    model.add(layers.Input({noise_dim}))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))   
    
    model.add(layers.Dense(256, activation="leaky_relu"))    
    model.add(layers.Dense(feature_dim))
    model.compile()
    
    print(model.output_shape)
    assert model.output_shape == (None, feature_dim)               

    return model


def ConvDiscriminator(feature_dim):
    model = Sequential()

    model.add(layers.Input(shape={feature_dim}))
    model.add(layers.Reshape([feature_dim, 1]))

    model.add(layers.Conv1D(kernel_size=15,
                            filters=256,
                            activation='leaky_relu'))
    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(kernel_size=15,
                            filters=128))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(layers.MaxPool1D())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(1))

    model.compile()

    return model
