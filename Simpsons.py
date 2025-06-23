import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
#import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD



IMG_SIZE = (80, 80)
channels = 1
char_path = r'D:\Projects\OpenCV\Simpsons project\Simpsons-project\simpsons character dataset\simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))
    #grab all the folders in the path and find number of images in each folder(each simpson character)

char_dict = caer.sort_dict(char_dict, descending = True)
# print(f'characters',char_dict)

#now adding characters
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
print(f'The characters are as follows:',characters)
    
train = caer.preprocess_from_dir(char_path, characters, channels = channels, IMG_SIZE=IMG_SIZE, isShuffle = True)
print(f'The length of the training data is:',len(train))

featureSet, labels = caer.sep_train(train, IMG_SIZE = IMG_SIZE)
#Seperate training set into features and labels

from tensorflow.keras.utils import to_categorical
#to_categorical is used to convert class labels (integers) into one-hot encoded vectors.
#normalize the featureset (0,1)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio = .2 )
#20 percent goes to validation set and 80 percent goes to the training


del train
del featureSet
del labels

gc.collect()

BATCH_SIZE = 32
EPOCHS = 20


#image data generator
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size = BATCH_SIZE)

#model creation
#output_dim = len(characters)
output_dim = 10
#output_dim is the number of classes we have

w, h = IMG_SIZE[:2]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w, h,channels)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

# Output Layer
model.add(Dense(output_dim, activation='softmax'))

model.summary()


# Training the model

optimizer = SGD(learning_rate=0.001, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen,
                    steps_per_epoch = len(x_train) // BATCH_SIZE, epochs = EPOCHS,
                    validation_data = (x_val, y_val), validation_steps = len(y_val) // BATCH_SIZE,
                    callbacks = callbacks_list)


# Testing

test_path = r'D:\Projects\OpenCV\Simpsons project\Simpsons-project\simpsons character dataset\kaggle_simpson_testset\kaggle_simpson_testset\bart_simpson_31.jpg'

img = cv.imread(test_path)

plt.imshow(img)
plt.show()

def prepare(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE)
    image = caer.reshape(image, IMG_SIZE, 1)
    return image

predictions = model.predict(prepare(img))

# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])
# Displaying the prediction