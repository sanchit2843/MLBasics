# Importing dependencies
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#loading dataset
train = pd.read_csv('train.csv')
X_train = train.iloc[0:40000,1:785].values
Y_train = train.iloc[0:40000,0:1].values
X_test = train.iloc[40000:42000,1:785].values
Y_test = train.iloc[40000:42000,0:1].values

#one hot encoding labels

onehotencoder = OneHotEncoder(categorical_features=[0])
Y_train = onehotencoder.fit_transform(Y_train).toarray()

#creating 4-d varriables for image data
Xtrain = [1] * 31360000
Xtrain = np.reshape(Xtrain,(40000,28,28,1))
Xtest = [1] * 1568000
X1 = [1]*784

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

Xtest = np.reshape(Xtest,(2000,28,28,1))

for i in range(0,40000):
    X1 = X_train[i,:]
    X1 = np.reshape(X1,(28,28,1))
    Xtrain[i,:,:] = X1
for i in range(0,2000):
    X1 = X_test[i,:]
    X1 = np.reshape(X1,(28,28,1))
    Xtest[i,:,:] = X1
    
#creating sequential cnn model
classifier = Sequential()
classifier.add(Conv2D(32,3,3,input_shape = (28,28,1),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(16,3,3,input_shape = (28,28,1),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 800, init = 'uniform' , activation ='relu',input_dim = 784))
classifier.add(Dense(output_dim = 400, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 200, init = 'uniform' , activation ='relu'))

classifier.add(Dense(output_dim = 100, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 50, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 20, init = 'uniform' , activation ='relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform' , activation ='sigmoid'))

#compiling model
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' , metrics = ['accuracy'])

#fitting cnn to training set
classifier.fit(X_train,Y_train, batch_size=10 , nb_epoch = 10)

#predicting results
y_pred = classifier.predict(X_test)

# reversing process of one hot encoding
for i in range(0,2000):
    for j in range(0,10):
        if(y_pred[i,j]<0.5):
            y_pred[i,j]=0
        if(y_pred[i,j]>0.5):
            y_pred[i,j]=1
for i in range(0,2000):
    for j in range(0,10):
        if(y_pred[i,j]==1):
            y_pred[i,0] = j

y_pred = np.delete(y_pred,np.s_[1:10],axis = 1)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

#getting accuracy score
from sklearn.metrics import accuracy_score
a = accuracy_score(Y_test,y_pred)

#checking model on random images
image = cv2.imread('download.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image1 = cv2.resize(image,(28,28))
for i in range(0,28):
    for j in range(0,28):
        image1[i,j] = 255-image1[i,j]
#image1 = np.reshape(image1,(1,784))
image1 = np.reshape(image1,(1,28,28,1))
y_pred1 = classifier.predict(image1)

#For importing model from saved weights
'''from keras.models import model_from_json
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")
from keras.models import model_from_json
json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])'''