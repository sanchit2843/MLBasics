import librosa
import cv2
import numpy as np
import os
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Dataset:
    def __init__(self,song_folder = None,new_folder=None, csv_location = None):
        self.song_folder = song_folder
        self.new_folder = new_folder
        self.csv_location = csv_location
    
    def remove_unlabelled(self):
        track = pd.read_csv(self.csv_location)
        y = track.iloc[:,1:2].values
        ID = track.iloc[:,0:1].values
        y = y[2:,:]
        ID = ID[2:,:]
        
        label = {}
        for i,j in zip(ID,y):
          label[str(i[0])] = j[0]
      
        for image in os.listdir(self.new_folder):
          path = os.path.join(self.new_folder,image)
          file_name = image
          file_name = file_name[:-6]
          file_name = "".join(file_name)
          genre = label.get(file_name,'notfound')
          if genre == 'notfound':
            os.remove(path)
            
    def chop_image(self, img_path, out_path):
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        w = width // 10
        for i in range(0,10):
          temp = img[:, (i*w):((i+1)*w), :]
          temp = cv2.resize(temp,(128,128))
          cv2.imwrite((out_path+'/spec_{}.png'.format(i)), temp)
    #mel spectogram for 1 audio
    def spec_create(self, in_path,out_path):
        x,sr = librosa.load(in_path,sr=44100,mono=True)
        fig = plt.Figure(figsize=(36,15))   
        #canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(x))),sr=sr,ax=ax)
        output = out_path + '/test_img.jpg'
        fig.tight_layout()
        fig.savefig(output)
        self.chop_image(output, out_path)
   
    #Driver Program   
    def create_data(self):
        for i in tqdm(os.listdir(self.song_folder)):
            folder = os.path.join(self.song_folder,i)
            sub_folder = os.listdir(folder)
            for track in sub_folder:
                path = os.path.join(folder,track)
                track = list(track)
                track = track[:-4]
                track = "".join(track)
                track = int(track)
                track = str(track)
                output = os.path.join(self.new_folder,track)
                try:
                    self.spec_create(path,output)
                except:
                    print('Corrupt file {}'.format(path))
        self.remove_unlabelled()
