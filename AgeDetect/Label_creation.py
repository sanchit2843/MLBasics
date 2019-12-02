from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='0BxYys69jI14kYVM3aVhKS1VhRUk',
                                    dest_path='/content/UTKFace.tar.gz')
import tarfile
tar = tarfile.open("/content/UTKFace.tar.gz")
tar.extractall()
tar.close()
import numpy as np
label = np.ones((23708,2),int)
a = os.listdir('/content/UTKFace')
for i in range(len(a)):
  b = str(a[i])
  if(ord(b[1])==95):
     label[i,0] = int(b[0])
     label[i,1] = int(b[2])
  elif(ord(b[2])==95):
     label[i,0] = int(b[0:2])
     label[i,1] = int(b[3])
  elif(ord(b[3])==95):
     label[i,0] = int(b[0:3])
     label[i,1] = int(b[4])
a = np.array(a)
a = np.reshape(a,(23708,1))
c = np.concatenate((a,label),axis = 1)
import pandas as pd
data = pd.DataFrame(c)
data.to_csv('label.csv')