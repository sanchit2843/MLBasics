import os
os.system('pip install wget')
os.system('pip install patool')
os.system('pip install pyunpack')
os.system('pip install -i https://test.pypi.org/simple/ supportlib')
import shutil
import os
import wget
from tqdm import tqdm
import supportlib.gettingdata as getdata

#upload kaggle.json file
getdata.kaggle()
os.system('kaggle competitions download -c planet-understanding-the-amazon-from-space')
from pyunpack import Archive
Archive('/content/train-tif-v2.tar.7z').extractall('/content')
getdata.tarextract('/content/train-tif-v2.tar')
getdata.zipextract('/content/train_v2.csv.zip')
