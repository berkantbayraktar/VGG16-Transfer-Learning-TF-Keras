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


def plot_trainset(x_train, x_test):
    print('Train: X=%s, y=%s' % (x_train.shape, x_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, x_test.shape))
    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()


class VGG16:
    def __init__(self, number_of_classes):
        self.model = None
        self.number_of_classes = number_of_classes
        pass

    def create_model(self):
        pass


if __name__ == "__main__":

    # with include_top = False, you can pull just CNNs not FC layers. We use VGG19 as backbone
    base_model = VGG19(weights="imagenet", include_top=False, input_shape= (100,100,3))
    base_model.summary()

    # extend the network with FC layers
    new_model = extend_network(base_model=base_model, number_of_class=131)
    new_model.summary()

    #model.predict()
    #plot_trainset(x_train, y_train)
