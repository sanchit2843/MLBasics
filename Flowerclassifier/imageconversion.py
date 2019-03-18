import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
#class 1
im = [1]*2694905
image1 = [1]*385728
image1 = np.reshape(image1,(492,784))
im = np.reshape(im,(3433,785))
for i in range (1,492):
    a = '1 (' + str(i) + ').jpg'
    image = cv2.imread(a)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(28,28))
    image = np.reshape(image,(1,784))
    image1[i] = image
image = cv2.imread('1 (492).jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.resize(image,(28,28))
image = np.reshape(image,(1,784))
image1[0] = image
Y = [0]*492
Y = np.reshape(Y,(492,1))
image1 = np.concatenate((image1,Y),axis = 1)
im[0:492,:] = image1

#class 2
image2 = [1]*377104
image2 = np.reshape(image2,(481,784))
for i in range (1,481):
    a = '1 (' + str(i) + ').jpg'
    image = cv2.imread(a)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(28,28))
    image = np.reshape(image,(1,784))
    image2[i] = image
image = cv2.imread('1 (481).jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.resize(image,(28,28))
image = np.reshape(image,(1,784))
image2[0] = image
Y = [3]*481
Y = np.reshape(Y,(481,1))
image2 = np.concatenate((image2,Y),axis = 1)
im[2952:3433,:] = image2
im = pd.DataFrame(im)
im.to_csv('1.csv')