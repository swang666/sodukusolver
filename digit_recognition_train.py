import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import gc
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

'''
puzzle = ps.puzzle_process('puzzles/8.jpg')
path = "D:/UCLA/project/soduku/trainimage/"

currentDT = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

for i in range(0,9):
    for j in range(0,9):       
        crop_img = puzzle[i * 40: (i+1)*40 - 1, j * 40: (j+1)*40 - 1]
        filename = path + currentDT + "_" + str(i*9 + j) + ".jpg"
        cv2.imwrite(filename, crop_img)

'''
# https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9

# import train images
path = "D:/UCLA/project/soduku/trainimage/"
keys = ['blank/', 'one/','two/','three/','four/','five/','six/','seven/','eight/','nine/']
blanks = [(path + keys[0] + "{}").format(i) for i in os.listdir(path + keys[0])]
ones = [(path + keys[1] + "{}").format(i) for i in os.listdir(path + keys[1])]
twos = [(path + keys[2] + "{}").format(i) for i in os.listdir(path + keys[2])]
threes = [(path + keys[3] + "{}").format(i) for i in os.listdir(path + keys[3])]
fours = [(path + keys[4] + "{}").format(i) for i in os.listdir(path + keys[4])]
fives = [(path + keys[5] + "{}").format(i) for i in os.listdir(path + keys[5])]
sixs = [(path + keys[6] + "{}").format(i) for i in os.listdir(path + keys[6])]
sevens = [(path + keys[7] + "{}").format(i) for i in os.listdir(path + keys[7])]
eights = [(path + keys[8] + "{}").format(i) for i in os.listdir(path + keys[8])]
nines = [(path + keys[9] + "{}").format(i) for i in os.listdir(path + keys[9])]

train_imgs = blanks + ones + twos + threes + fours + fives + sixs + sevens + eights + nines 
random.shuffle(train_imgs)
# clear garbage
del blanks 
del ones 
del twos
del threes
del fours
del fives
del sixs
del sevens
del eights
del nines

gc.collect()

def read_and_process_image(list_of_images):
    X = []
    y = []

    for image in list_of_images:
        img = cv2.resize(cv2.imread(image, cv2.COLOR_BGR2GRAY), (150,150), interpolation = cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        if 'blank' in image:
            y.append(0)
        elif 'one' in image:
            y.append(1)
        elif 'two' in image:
            y.append(2)
        elif 'three' in image:
            y.append(3)
        elif 'four' in image:
            y.append(4)
        elif 'five' in image:
            y.append(5)
        elif 'six' in image:
            y.append(6)
        elif 'seven' in image:
            y.append(7)
        elif 'eight' in image:
            y.append(8)
        elif 'nine' in image:
            y.append(9)
    
    return X, y

X, y = read_and_process_image(train_imgs)
print(X[0].shape)
X = np.array(X)
y = np.array(y)
# split the data into train and test set

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

del X
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)

batch_size = 32

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (150,150, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(10, activation='softmax'))

# use the RMSprop optimizer with a learning rate of 0.0001
# use sparse_categorical_crossentropy loss because 10 classes
model.compile(loss ='sparse_categorical_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['acc'])

# create the augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1,
                                  shear_range = 0.1,
                                  zoom_range = 0.1)
val_datagen = ImageDataGenerator(rescale=1./255)

# create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val,y_val,batch_size=batch_size)

# training
history = model.fit_generator(train_generator,
                                steps_per_epoch = ntrain // batch_size,
                                epochs = 64,
                                validation_data = val_generator,
                                validation_steps = nval // batch_size)

model.save('models/digit_rec_model.h5')