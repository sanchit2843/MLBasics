from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import cv2

#import tensorflow as tf
#initializing ANN
classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape = (28,28,3),activation = 'relu'))
#pooling 
classifier.add(MaxPooling2D(pool_size=(2,2)))
#flatten
classifier.add(Flatten())
#compiling
classifier.add(Dense(output_dim = 800, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 400, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 200, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 50, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 5, init = 'uniform' , activation ='sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' , metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#y_pred = classifier.predict()
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'E:/Computer/Computervision/flowers',
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'E:/Computer/Computervision/New Folder (2)',
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')
classifier.fit_generator(
        train_generator,
        steps_per_epoch=4114,
        epochs=7,
        validation_data=validation_generator,
        validation_steps=210)
from keras.models import model_from_json
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")
#loading model
import numpy as np
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from keras.preprocessing import image
image1 = image.load_img('E:/Computer/Computervision/flowers/download.png',target_size = (28,28))
image1 = image.img_to_array(image1)
image1 = np.expand_dims(image1,axis=0)
y_pred1 = loaded_model.predict(image1)