import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
ses1 = os.listdir('./1st_session/raw_data')
for i in ses1:
  path = os.path.join('./2nd_session/raw_data',i)
  try:
    files = os.listdir(path)
  except:
    print('not directory')
  for j in files:
    try:
      src = os.path.join(path,j)
      j = list(j)
      q = int(j[1])
      q = q+6
      q = str(q)
      j[1] = q
      j = ''.join(j)
      des = os.path.join(path,j)
      os.rename(src,des)
    except:
      print('notdirectory')
import shutil
for i in ses1:
  src = os.path.join('./2nd_session/raw_data',i)
  des = os.path.join('./1st_session/raw_data',i)
  try:
    files = os.listdir(path)
  except:
    print('not directory')
  for j in files:
    try:
      src1 = os.path.join(src,j)
      des1 = os.path.join(des,j)
      shutil.copyfile(src1,des1)
    except:
      print('notdirectory')




def preprocess(image):
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(image,M,(cols,rows))
    kernel1 = np.ones((4,20))
    kernel1[0:2,:] = -1
    kernel2 = np.ones((4,20))
    kernel2[2:4,:] = -1
    dst1 = cv2.filter2D(dst,-1,kernel1)
    dst2 = cv2.filter2D(dst,-1,kernel2)
    _,dst3 = cv2.threshold(dst1,250,255,0)
    
    a,b = list(),list()
    rows,cols = dst3.shape
    
    for i in range(rows):
        for j in range(cols):
            if(dst3[i,j]==255):
                a.append(i)
                b.append(j)
    rmin = min(a)
    dst2 = dst2[rmin+50:]
    _,dst4 = cv2.threshold(dst2,250,255,0)
    rows,cols = dst4.shape
    
    for i in range(rows):
        for j in range(cols):
            if(dst4[i,j]==255):
                a.append(i)
                b.append(j)
    
    rmax = min(a)+50+rmin
    kernel3 = np.ones((20,4))
    kernel3[:,0:2] = -1
    kernel4 = np.ones((20,4))
    kernel4[:,2:4] = -1
    dst5 = cv2.filter2D(dst,-1,kernel3)
    dst6 = cv2.filter2D(dst,-1,kernel4)
    s = dst5.sum(axis=0)
    s = s.argsort()
    l = len(s)
    cmin = s[l-1]
    s = dst6.sum(axis=0)
    s = s.argsort()
    cmax = s[l-1]
    dst = dst[rmin-5:rmax+20,cmin-5:cmax-30]
    dst = cv2.resize(dst,(153,63))
    return dst

import os
import cv2
import numpy as np
!mkdir processeddata
ses1 = os.listdir('./1st_session/raw_data')

from tqdm import tqdm
for i in tqdm(ses1):
  path = os.path.join('./1st_session/raw_data',i)
  path1 = os.path.join('./processeddata',i)
  try:
    os.mkdir(path1)
    files = os.listdir(path)
  except:
    a = None
  for j in files:
    src = os.path.join(path,j)
    try:
        image = cv2.imread(src,0)
        image = preprocess(image)
        des = os.path.join(path1,j)
        cv2.imwrite(des,image)
    except:
        a = None