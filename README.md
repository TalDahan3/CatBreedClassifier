# CatBreedClassifier
This program is a demo program for using a Convolutional Nerual Net to classify cat breed images.
This program uses the Keras framework with the Tensorflow backend.

# Training
The training procedure is as follows:
1. Download cats images from Imagenet urls - training set.
2. Once downloaded the training process starts.
3. Every epoch the images get augmented.
4. The process repeats itself for the requested amount of epchos.
5. Only the latest weight of the CNN will be saved.

# Accuracy
The accuaracy procedure is as follows:
1. Download cats images from Imagenet urls - validation set.
2. Predict the cats breeds - feeding the image to the CNN performing a forward pass.
3. Printing the success rate of the classifications.

# Usage
Training - 
In order to train the CNN from scratch run the program and press t. 
In case you want to train from previous weights go to __main and call the training function
with the first variables as the path to the weights instead of None.
To determine the number of epochs and batch size call the training program with suited veriables.

Accuracy-
In order to calculate the success rate of the classifications run the program and press a.

