#plotting contours for multiple digit recognition
import cv2
import numpy as np

def digit_localization(test):
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#binarize image
    ret,thresh = cv2.threshold(test,127,255,cv2.THRESH_BINARY)
#invert color of image
    inverted = abs(255-thresh)
#Dilate image
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(inverted,kernel,iterations = 1)
    im2, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    d = list()
    for c in contours:
    # get the bounding rect
        x,y,w,h = cv2.boundingRect(c)
        t = (x,y,w,h)
        d.append(t)
    return d
def crop_digits(image):
    image1 = cv2.imread(image)
    e = digit_localization(image1)
    a = len(e)
    image2 = list()
    for i in range(a):
        (x,y,w,h) = e[i]
        img = image1[y:y+h,x:x+w]
        image2.append(img)
    return image2

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dropout
from keras.layers import BatchNormalization,Average
import collections

classifier = Sequential()
classifier.add(Conv2D(64,5,5,input_shape = (28,28,1),activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(128,5,5,activation = 'relu'))
classifier.add(Dropout(0.2)
classifier.add(BatchNormalization())
classifier.add(Flatten())
classifier.add(Dense(output_dim = 500, init = 'uniform' , activation ='relu',input_dim = 784))
#classifier.add(Dense(output_dim = 1000, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform' , activation ='softmax'))
#compiling model
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy' , metrics = ['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=1, min_lr=0.00000001)
callbacks_list = [reduce_lr]
#fitting cnn to training set
classifier.fit_generator(datagen.flow(X, Y, batch_size=32),steps_per_epoch=len(X) / 32,nb_epoch = 20 ,callbacks = callbacks_list)

model  = load_weights('classifier.h5')
plt.imshow(image1)
image1 = image[0]
image1 = cv2.resize(image1,(28,28))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = 255-image1

image1 = np.reshape(image1,(1,28,28,1))
model.predict(image1)

