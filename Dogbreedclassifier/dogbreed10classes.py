#dog breed classifier on 10 classes of stanford dataset with folder creation
#link for dataset - http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar


#importing dependencies
!pip install requests
!pip install tqdm
import requests
!pip install wget
import wget
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.applications import VGG19


# downloading dataset
url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'  
wget.download(url, '/content')

#extract dataset
import tarfile
tar = tarfile.open("C:/Users/sanchitcloud1/Downloads/images.tar")
tar.extractall()
tar.close()

#getting filenames
import os
a = os.listdir('C:/Users/sanchitcloud1/Downloads/Images')

#copying data from main folder to new folder and creating new directory
import shutil
src = 'C:/Users/sanchitcloud1/Downloads/Images'
des = 'C:/Users/sanchitcloud1/Downloads/classes10'
os.makedirs('C:/Users/sanchitcloud1/Downloads/classes10')
print(src)
print(des)

src1 = [1]*10
#loop to rename folders similar to classes
for i in range(120):
  b = a[i]
  b = str(b)
  b = list(b)

  size = len(b)
  b = b[10:size]
  res = ''.join(b)
  res = res.lower()
  src = os.path.join('C:/Users/sanchitcloud1/Downloads/Images',a[i])
  dest = os.path.join('C:/Users/sanchitcloud1/Downloads/Images',res)
  os.rename(src,dest)

#sorting classes according to dictionary
a = os.listdir('C:/Users/sanchitcloud1/Downloads/Images/')
a.sort()
print(a)
#moving files

for i in range(10):
  src1[i] = os.path.join(src,a[i])
  print(src1[i])
  shutil.move(src1[i],des)
  
#data augmentation for train and test data using keras datagenerator

datagen = ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=0.2,
    
    height_shift_range=0.16,
    shear_range=0.1,
    zoom_range=.1,
    horizontal_flip=True,
    vertical_flip=True)
train_generator = datagen.flow_from_directory(
        'C:/Users/sanchitcloud1/Downloads/classes10',
        target_size=(139, 139),
        batch_size=1,
        class_mode='categorical')
test_generator = datagen.flow_from_directory(
        'C:/Users/sanchitcloud1/Downloads/classes10test',
        target_size=(139, 139),
        batch_size=1,
        class_mode='categorical')

# creating variables for training and test data

x = np.ones((1850,139,139,3),dtype = int)
y = np.ones((1850,10))
X_test = np.ones((940,139,139,3),dtype = int)
Y_test = np.ones((940,10))

#getting images in 4-d array created
for i in tqdm(range(0,1850)):
    x[i,:,:,:],y[i,:] = train_generator.next()
for i in tqdm(range(0,940)):
    X_test[i,:,:,:],Y_test[i,:] = test_generator.next()

#transfering weights for vgg architecture    
vgg_conv = VGG19(include_top=False, weights='imagenet',input_shape=(139,139,3))

#declaring model
modelnn = Sequential()
modelnn.add(vgg_conv)  
modelnn.add(Flatten())
modelnn.add(Dense(500, activation='relu'))
modelnn.add(BatchNormalization())
modelnn.add(Dropout(0.5))
modelnn.add(Dense(250, activation='relu'))
modelnn.add(BatchNormalization())
modelnn.add(Dropout(0.5))
modelnn.add(Dense(10, activation='softmax'))


#declaring callbacks
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False)
filepath="weights-improvement-{epoch:02d}-{acc:.2f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list = [reduce_lr,tensorboard,checkpoint]
#compiling model
modelnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#fit model
modelnn.fit_generator(datagen.flow(x, y, batch_size=32),steps_per_epoch=len(x) / 32, epochs=100,validation_data = (X_test,Y_test),callbacks = callbacks_list)

#predicting results
y_pred = modelnn.predict(X_test)

#reversing process of one hot encoder to get final result
for i in range(940):
    for j in range(10):
        if(y_pred[i,j] == 1):
            y_pred[i,0] = j
        if(Y_test[i,j] == 1):
            Y_test[i,0] = j
Y_test = np.delete(Y_test,np.s_[1:10],axis  =1)
y_pred = np.delete(y_pred,np.s_[1:10],axis  =1)

#getting accuracy
from sklearn.metrics import accuracy_score,confusion_matrix
a = accuracy_score(Y_test,y_pred)
