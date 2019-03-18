!pip install tqdm
!pip install kaggle

from googleapiclient.discovery import build
import io, os
from googleapiclient.http import MediaIoBaseDownload
from google.colab import auth
auth.authenticate_user()
drive_service = build('drive', 'v3')
results = drive_service.files().list(
        q="name = 'kaggle.json'", fields="files(id)").execute()
kaggle_api_key = results.get('files', [])
filename = "/content/.kaggle/kaggle.json"
os.makedirs(os.path.dirname(filename), exist_ok=True)
request = drive_service.files().get_media(fileId=kaggle_api_key[0]['id'])
fh = io.FileIO(filename, 'wb')
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print("Download %d%%." % int(status.progress() * 100))
os.chmod(filename, 600)

!kaggle competitions download -c dog-breed-identification

import os
print(os.listdir('/content/'))

import zipfile
zip_ref = zipfile.ZipFile('/content/labels.csv.zip', 'r')
zip_ref.extractall('/content/')
zip_ref.close()

import zipfile
zip_ref = zipfile.ZipFile('/content/train.zip', 'r')
zip_ref.extractall('/content/')
zip_ref.close()

from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=0.1,
    
    height_shift_range=0.1,
    shear_range=0.,
    zoom_range=.1,
    horizontal_flip=True,
    vertical_flip=True)

import pandas as pd
label = pd.read_csv('/content/labels.csv')
id = label['id'].values
breed = label['breed'].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb = LabelEncoder()
breed = lb.fit_transform(breed)

'''import numpy as np
breed = np.reshape(breed,(10222,1))
ohe = OneHotEncoder(categorical_features = [0])
breed = ohe.fit_transform(breed).toarray()'''

path = '/content/train/'
c = '.jpg'
from keras.preprocessing import image
from tqdm import tqdm

x = [1]*125607936

import numpy as np
x = np.reshape(x,(10222,64,64,3))

import cv2
for i in tqdm(range(0,10222)):
  a = str(id[i])
  b = path + a + c
  image = cv2.imread(b)
  image = cv2.resize(image,(64,64))
  
  x[i,:,:,:] = image

import os
os.remove('/content/train.zip')
os.remove('/content/test.zip')
os.remove('/content/labels.csv.zip')

x1 = [1]*18333696
x1 = np.reshape(x1,(1492,64,64,3))

breed2 = [1]*1492

j=0
for i in range(10222):
  if(breed[i]<16):
    breed2[j] = breed[i]
    x1[j,:,:,:] = x[i,:,:,:]
    j=j+1
print(breed2)

import numpy as np
breed2 = np.reshape(breed2,(1492,1))
ohe = OneHotEncoder(categorical_features = [0])
breed2 = ohe.fit_transform(breed2).toarray()

np.shape(breed2)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.applications import VGG19
vgg_conv = VGG19(include_top=False, weights='imagenet',input_shape=(64,64,3))

modelnn = Sequential()
    modelnn.add(vgg_conv)  
    
#     modelnn.add(Convolution2D(64, 3, 3, input_shape=(64,64,3), activation='relu', border_mode='same'))
#     modelnn.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#     modelnn.add(MaxPooling2D(pool_size=(2, 2)))
#     modelnn.add(BatchNormalization())
#     modelnn.add(Dropout(0.2))
#     modelnn.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#     modelnn.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
#     modelnn.add(BatchNormalization())
#     # modelnn.add(MaxPooling2D(pool_size=(2, 2)))
#     modelnn.add(Dropout(0.5))
#     modelnn.add(BatchNormalization())
    modelnn.add(Flatten())
    
    modelnn.add(Dense(500, activation='relu'))
    modelnn.add(BatchNormalization())
    modelnn.add(Dropout(0.5))
    modelnn.add(Dense(250, activation='relu'))
    modelnn.add(BatchNormalization())
    modelnn.add(Dropout(0.5))
    modelnn.add(Dense(16, activation='softmax'))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    
#data augmentationbatch_size = 64

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [reduce_lr]
modelnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

modelnn.summary()
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x1,breed2,test_size = 0.2)

modelnn.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),steps_per_epoch=len(X_train) / 32, epochs=100,validation_data = (X_test,Y_test),callbacks = callbacks_list)

model_json = modelnn.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
modelnn.save_weights("classifier.h5")
print("Saved model to disk")
