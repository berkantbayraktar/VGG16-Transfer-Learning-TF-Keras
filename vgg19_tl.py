import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras import utils
from tensorflow.keras import datasets
from tensorflow.keras import preprocessing
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import numpy as np
import os
'''
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv4 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv4 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv4 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
'''


def extend_network(base_model, number_of_class):
    # don't train existing weights of Convolution Layers
    for layer in base_model.layers:
        layer.trainable = False

    # prediction layers
    flatten = layers.Flatten(name='flatten')
    fc_1 = layers.Dense(4096, activation='relu', name='fc1')
    fc_2 = layers.Dense(4096, activation='relu', name='fc2')
    fc_3 = layers.Dense(number_of_class, activation='softmax', name='predictions')

    # create new model object

    new_model = models.Sequential(name="new_model")

    # add prediction layers to new model object
    new_model.add(base_model)
    new_model.add(flatten)
    new_model.add(fc_1)
    new_model.add(fc_2)
    new_model.add(fc_3)

    # define optimizer
    optimizer = optimizers.Adam(learning_rate=0.001)

    # compile new model
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return new_model


def data_generator(network_size):
    generator = preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        vertical_flip=True,
        rotation_range=20
    )
    return generator


def train_generator(data_generator, path, network_size):
    train_gen = data_generator.flow_from_directory(
        path,
        batch_size=32,
        target_size=network_size
    )

    return train_gen


def test_generator(data_generator, path, network_size):
    test_gen = data_generator.flow_from_directory(
        path,
        batch_size=32,
        target_size=network_size
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
    base_model = VGG19(weights="imagenet", include_top=False, input_shape= (100,100,3))
    base_model.summary()

    # extend the network with FC layers
    new_model = extend_network(base_model=base_model, number_of_class=number_of_classes)
    new_model.summary()

    data_gen = data_generator(network_size)
    # get current path
    cwd = os.getcwd()

    train_gen = train_generator(data_gen, os.path.join(cwd, "data/fruits-360/Training"), network_size)
    test_gen = test_generator(data_gen, os.path.join(cwd, "data/fruits-360/Test"), network_size)

    res = new_model.fit_generator(
        generator=train_gen,
        validation_data=test_gen,
        epochs=10
    )

    plot_graphs(res)