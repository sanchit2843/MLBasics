

!pip install tqdm

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

!pip install kaggle
!kaggle datasets download -d moltean/fruits

path = '/content/.kaggle/datasets/moltean/fruits/fruits-360/Training'

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
train_generator = datagen.flow_from_directory(
        '/content/.kaggle/datasets/moltean/fruits/fruits-360/Training',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

x = [1]*464928768



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import cv2
import numpy as np

classifier = Sequential()
classifier.add(Conv2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))
classifier.add(Conv2D(16,3,3,input_shape = (64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(16,3,3,input_shape = (32,32,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(8,3,3,input_shape = (16,16,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(8,3,3,input_shape = (8,8,3),activation = 'relu'))

classifier.add(Flatten())

#compiling
classifier.add(Dense(output_dim = 150, init = 'uniform' , activation ='relu',input_dim = 192))
classifier.add(Dense(output_dim = 75, init = 'uniform' , activation ='softmax'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' , metrics = ['accuracy'])
classifier.fit_generator(train_generator,steps_per_epoch=37836,epochs=10)