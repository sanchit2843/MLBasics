!pip install requests
!pip install tqdm
import requests
!pip install wget
import wget

print('Beginning file download with wget module')

url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'  
wget.download(url, '/content')
import tarfile
tar = tarfile.open("/content/images.tar")
tar.extractall()
tar.close()

import os
a = os.listdir('/content/Images/')
print(a)
#os.makedirs('/content/classes10')
import shutil
src = '/content/Images'
des = '/content/classes10'
src1 = [1]*10
for i in range(120):
  b = a[i]
  b = str(b)
  b = list(b)

  size = len(b)
  b = b[10:size]
  res = ''.join(b)
  res = res.lower()
  src = os.path.join('/content/Images',a[i])
  dest = os.path.join('/content/Images',res)
  os.rename(src,dest)

#shutil.rmtree('/content/Images', ignore_errors=False, onerror=None)
a = os.listdir('/content/Images/')
a.sort()
print(a)
des = '/content/classes10'

src = '/content/Images'
#shutil.rmtree('/content/Images', ignore_errors=False, onerror=None)

os.makedirs('/content/classes10')
print(os.listdir('/content'))
print(src)
print(des)

for i in range(10):
  src1[i] = os.path.join(src,a[i])
  print(src1[i])
  shutil.move(src1[i],des)

from keras.preprocessing.image import ImageDataGenerator
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
        '/content/classes10',
        target_size=(139, 139),
        batch_size=1,
        class_mode='categorical')
import numpy as np

x = np.ones((1850,139,139,3),dtype = int)
y = np.ones((1850,10))
from tqdm import tqdm

for i in tqdm(range(0,1850)):
    x[i,:,:,:],y[i,:] = train_generator.next()

!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
#!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
#!apt-get update -qq 2>&1 > /dev/null
#!apt-get -y install -qq google-drive-ocamlfuse fuse
!wget https://launchpad.net/~alessandro-strada/+archive/ubuntu/google-drive-ocamlfuse-beta/+build/15331130/+files/google-drive-ocamlfuse_0.7.0-0ubuntu1_amd64.deb
!dpkg -i google-drive-ocamlfuse_0.7.0-0ubuntu1_amd64.deb
!apt-get install -f
!apt-get -y install -qq fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
!mkdir -p drive
!google-drive-ocamlfuse drive

print(os.listdir('/content/drive'))

from keras.callbacks import ReduceLROnPlateau,TensorBoard
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False)
callbacks_list = [reduce_lr]

from keras.models import model_from_json
import numpy as np
json_file = open('/content/drive/dogbreed10classes/classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/dogbreed10classes/latest.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model.summary()
loaded_model.fit_generator(datagen.flow(x,y,batch_size = 32),steps_per_epoch= 1850/ 32, epochs=100,callbacks = callbacks_list)

model_json = loaded_model.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("classifier.h5")
print("Saved model to disk")
#loading model
!pip install pydrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
file1 = drive.CreateFile({"mimeType": "text/csv"})
file1.SetContentFile("/content/classifier.h5")
file1.Upload()
#!pip install pydrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
file1 = drive.CreateFile({"mimeType": "text/csv"})
file1.SetContentFile("/content/classifier.json")
file1.Upload()