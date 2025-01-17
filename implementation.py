import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import io
import os
import matplotlib.pyplot as plt

test_x = np.loadtxt("Dataset/test_X.csv", delimiter = ',') # Loading the test dataset
test_y = np.loadtxt("Dataset/test_label.csv", delimiter = ',')

test_x = test_x.reshape(350, 28, 28, 1) # Reshaping it
test_x = test_x/255 # Changing the range of values according to train dataset

try: # If the trained model already exists, then load it
    model = load_model(os.path.join('./saved_models/', 'model.h5'))
    print('Model loaded successfully')
except OSError: # If not found then train it
    print('Weights not found!\nTraining the model...')
    train_x = np.loadtxt("Dataset/train_X.csv", delimiter = ',') # Loading the train dataset
    train_y = np.loadtxt("Dataset/train_label.csv", delimiter = ',')

    train_x = train_x.reshape(1000, 28, 28, 1) # Reshaping it
    train_x = train_x/255 # hanging the range of values to make it faster for training

    model = Sequential()

    model.add(Conv2D(32,(3,3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
    model.fit(train_x,
              train_y,
              epochs = 1000,
              batch_size = 1024)
    print('Model trained successfully!')
    model.save(os.path.join('./saved_models/', 'model.h5')) # Savw the model
    print('Model saved!')

model.evaluate(test_x, test_y) # Evaluate the accuracy of the model

itr = int(input('How many examples do you want to see? ')) # number of iterations for example

for i in range(itr):
    index = random.randint(0, len(test_y))
    plt.imshow(test_x[index, :])
    plt.show()
    prediction_y = model.predict(test_x[index, :].reshape(1, 28, 28, 1))
    predicted_value = np.argmax(prediction_y)
    print("The predicted digit is", predicted_value)
