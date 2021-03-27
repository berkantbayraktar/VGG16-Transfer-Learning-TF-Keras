import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras import utils
from tensorflow.keras import datasets
from tensorflow.keras import preprocessing
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
'''
Model: "new_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg19 (Model)                (None, 3, 3, 512)         20024384  
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0         
_________________________________________________________________
dropout (Dropout)            (None, 4608)              0         
_________________________________________________________________
predictions (Dense)          (None, 131)               603779    
=================================================================
Total params: 20,628,163
Trainable params: 603,779
Non-trainable params: 20,024,384
'''


def extend_network(base_model, number_of_class):
    # don't train existing weights of Convolution Layers
    for layer in base_model.layers:
        layer.trainable = False

    # prediction layers
    flatten = layers.Flatten(name='flatten')
    do_1 = layers.Dropout(0.1)
    fc_1 = layers.Dense(number_of_class, activation='softmax', name='predictions')

    # create new model object
    new_model = models.Sequential(name="new_model")

    # add prediction layers to new model object
    new_model.add(base_model)
    new_model.add(flatten)
    new_model.add(do_1)
    new_model.add(fc_1)

    # define optimizer
    optimizer = optimizers.Adam(learning_rate=0.001)

    # compile new model
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return new_model


def train_data_generator(network_size):
    generator = preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        rotation_range=20
    )
    return generator


def test_data_generator(network_size):
    generator = preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )
    return generator


def train_generator(data_generator, path, network_size):
    train_gen = data_generator.flow_from_directory(
        path,
        batch_size=32,
        target_size=network_size,
        shuffle=True
    )
    return train_gen


def test_generator(data_generator, path, network_size):
    test_gen = data_generator.flow_from_directory(
        path,
        batch_size=32,
        target_size=network_size,
        shuffle=True
    )

    return test_gen


def plot_graphs(res):
    # loss
    plt.plot(res.history['loss'], label='train loss')
    plt.plot(res.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(res.history['accuracy'], label='train acc')
    plt.plot(res.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # network size
    network_size = (100, 100)

    # number of classes
    number_of_classes = 131

    # with include_top = False, you can pull just CNNs not FC layers. We use VGG19 as backbone
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
    # base_model.summary()

    # extend the network with FC layers
    new_model = extend_network(base_model=base_model, number_of_class=number_of_classes)
    new_model.summary()

    # create test and data generators for training process
    train_data_gen = train_data_generator(network_size)
    test_data_gen = test_data_generator(network_size)

    # get current path
    cwd = os.getcwd()

    train_gen = train_generator(train_data_gen, os.path.join(cwd, "data/fruits-360/Training"), network_size)
    test_gen = test_generator(test_data_gen, os.path.join(cwd, "data/fruits-360/Test"), network_size)

    # if val_accuracy value does not increase for four epochs, then stop the training
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=4, verbose=0, mode='max')

    # save best model only
    mcp_save = ModelCheckpoint('model_{epoch:03d}-{val_accuracy:03f}.h5', save_best_only=True,
                               monitor='val_accuracy', mode='max')

    res = new_model.fit_generator(
        generator=train_gen,
        validation_data=test_gen,
        epochs=50,
        callbacks=[earlyStopping, mcp_save]
    )

    plot_graphs(res)
