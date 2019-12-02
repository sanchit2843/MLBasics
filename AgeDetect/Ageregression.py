import datacleaning
import cv2
import os
import numpy as np
from tqdm import tqdm
def read_image(image_directory):
    image_directory = image_directory
    image = np.ones((len(datacleaning.age),50,50))
    for i in tqdm(range(len(datacleaning.path1))):
        path2 = os.path.join(image_directory,datacleaning.path1[i,0])
        image[i,:,:] = cv2.imread(path2,0)
    return image
image = read_image('/content/croppedfaces')
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator( rotation_range=90, 
                 width_shift_range=0.1, height_shift_range=0.1, 
                 horizontal_flip=True) 
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import model_from_json

modelnn = Sequential()
modelnn.add(Convolution2D(32, 3, 3, input_shape=(50,50,1), activation='relu', border_mode='same'))
modelnn.add(MaxPooling2D(pool_size=(2, 2)))
modelnn.add(BatchNormalization())
modelnn.add(Dropout(0.2))
modelnn.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
modelnn.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
modelnn.add(MaxPooling2D(pool_size=(2, 2)))
modelnn.add(BatchNormalization())
modelnn.add(Flatten())
modelnn.add(Dense(1024, activation='relu'))
modelnn.add(BatchNormalization())
modelnn.add(Dropout(0.5))
modelnn.add(Dense(512, activation='relu'))
modelnn.add(BatchNormalization())
modelnn.add(Dropout(0.5))
modelnn.add(Dense(1, activation='linear'))
modelnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
modelnn.fit_generator(datagen.flow(image, datacleaning.age, batch_size=32),steps_per_epoch=len(image) / 32, epochs=20)