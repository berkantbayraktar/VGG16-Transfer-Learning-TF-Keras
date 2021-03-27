# VGG19-Transfer-Learning-TF-Keras

### VGG19 Network is used as a backbone for our architecture. Convolutional Layers are freezed and Top Fully Connected layers are discarded . 
### Classifier tested on [Fruit360 Dataset](https://www.kaggle.com/moltean/fruits) . !! 0.952 Final Accuraccy !! on Test set. 

# Software Requirements

tensorflow-gpu = 2.0.0
python = 3.7.10
cuda = 10.0
cudnn = 7.6.5

# Environment Setup

1- Setup your python environment

```
conda env create -f environment.yml
```

2- You can download Kaggle Fruit360 Dataset from [here](https://www.kaggle.com/moltean/fruits) . Note that the dataset includes 131 different classes. 

3- Put the dataset to the project folder shown as below

    └── VGG19-Transfer-Learning-TF-Keras
        
        └── data
            
            └── fruits-360
                
                ├── papers
                
                ├── Test
                
                ├── test-multiple_fruits
                
                └── Training
# Training Results

## Loss Graph: 

![graph_1](https://github.com/berkantbayraktar/VGG19-Transfer-Learning-TF-Keras/blob/master/loss_graph.png)

## Accuracy Graph:

![graph_2](https://github.com/berkantbayraktar/VGG19-Transfer-Learning-TF-Keras/blob/master/accuracy_graph.png)

## Training Steps:


Epoch 1/50
2116/2116 [==============================] - 169s 80ms/step - loss: 0.9059 - accuracy: 0.7963 - val_loss: 0.6214 - val_accuracy: 0.8594

Epoch 2/50
2116/2116 [==============================] - 168s 79ms/step - loss: 0.2493 - accuracy: 0.9402 - val_loss: 0.4560 - val_accuracy: 0.8962

Epoch 3/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.1558 - accuracy: 0.9601 - val_loss: 0.3662 - val_accuracy: 0.9081

Epoch 4/50
2116/2116 [==============================] - 168s 80ms/step - loss: 0.1216 - accuracy: 0.9663 - val_loss: 0.3326 - val_accuracy: 0.9211

Epoch 5/50
2116/2116 [==============================] - 169s 80ms/step - loss: 0.1033 - accuracy: 0.9704 - val_loss: 0.3567 - val_accuracy: 0.9163

Epoch 6/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.0855 - accuracy: 0.9748 - val_loss: 0.3500 - val_accuracy: 0.9247

Epoch 7/50
2116/2116 [==============================] - 166s 79ms/step - loss: 0.0785 - accuracy: 0.9761 - val_loss: 0.3020 - val_accuracy: 0.9301

Epoch 8/50
2116/2116 [==============================] - 166s 78ms/step - loss: 0.0697 - accuracy: 0.9793 - val_loss: 0.3173 - val_accuracy: 0.9290

Epoch 9/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.0677 - accuracy: 0.9791 - val_loss: 0.2883 - val_accuracy: 0.9338

Epoch 10/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.0620 - accuracy: 0.9810 - val_loss: 0.2666 - val_accuracy: 0.9413

Epoch 11/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.0585 - accuracy: 0.9817 - val_loss: 0.2715 - val_accuracy: 0.9372

Epoch 12/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.0538 - accuracy: 0.9828 - val_loss: 0.2519 - val_accuracy: 0.9444

Epoch 13/50
2116/2116 [==============================] - 168s 79ms/step - loss: 0.0511 - accuracy: 0.9836 - val_loss: 0.2469 - val_accuracy: 0.9446

Epoch 14/50
2116/2116 [==============================] - 166s 79ms/step - loss: 0.0512 - accuracy: 0.9834 - val_loss: 0.2910 - val_accuracy: 0.9390

Epoch 15/50
2116/2116 [==============================] - 166s 79ms/step - loss: 0.0484 - accuracy: 0.9848 - val_loss: 0.2979 - val_accuracy: 0.9372

Epoch 16/50
2116/2116 [==============================] - 164s 78ms/step - loss: 0.0467 - accuracy: 0.9848 - val_loss: 0.3037 - val_accuracy: 0.9409

Epoch 17/50
2116/2116 [==============================] - 166s 79ms/step - loss: 0.0460 - accuracy: 0.9848 - val_loss: 0.2719 - val_accuracy: 0.9471

Epoch 18/50
2116/2116 [==============================] - 165s 78ms/step - loss: 0.0426 - accuracy: 0.9863 - val_loss: 0.3360 - val_accuracy: 0.9372

Epoch 19/50
2116/2116 [==============================] - 166s 78ms/step - loss: 0.0413 - accuracy: 0.9863 - val_loss: 0.2720 - val_accuracy: 0.9520

Epoch 20/50
2116/2116 [==============================] - 165s 78ms/step - loss: 0.0422 - accuracy: 0.9865 - val_loss: 0.3170 - val_accuracy: 0.9421

Epoch 21/50
2116/2116 [==============================] - 165s 78ms/step - loss: 0.0414 - accuracy: 0.9863 - val_loss: 0.2532 - val_accuracy: 0.9498

Epoch 22/50
2116/2116 [==============================] - 164s 78ms/step - loss: 0.0392 - accuracy: 0.9872 - val_loss: 0.3202 - val_accuracy: 0.9414

Epoch 23/50
2116/2116 [==============================] - 165s 78ms/step - loss: 0.0374 - accuracy: 0.9875 - val_loss: 0.2577 - val_accuracy: 0.9526

Epoch 24/50
2116/2116 [==============================] - 165s 78ms/step - loss: 0.0385 - accuracy: 0.9877 - val_loss: 0.3131 - val_accuracy: 0.9440

Epoch 25/50
2116/2116 [==============================] - 167s 79ms/step - loss: 0.0363 - accuracy: 0.9881 - val_loss: 0.2842 - val_accuracy: 0.9497

Epoch 26/50
2116/2116 [==============================] - 166s 78ms/step - loss: 0.0350 - accuracy: 0.9883 - val_loss: 0.2727 - val_accuracy: 0.9498

Epoch 27/50
2116/2116 [==============================] - 166s 78ms/step - loss: 0.0367 - accuracy: 0.9883 - val_loss: 0.3312 - val_accuracy: 0.9380
