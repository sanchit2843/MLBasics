# MusicGenreClassification

Music Genre classification is an important problem for the services like spotify etc. We have tackled this problem by creating the mel spectograms of the data and converting them into images. The dataset used was fma_small which can be found [here](https://github.com/mdeff/fma). Spectograms were created using [librosa](https://librosa.github.io/librosa/). Preprocessed images can be downloaded using this [link](https://drive.google.com/open?id=1SKW6aNswBzWhG-LylVopaHTgz60mBLnx) and [labels](https://drive.google.com/file/d/1tIxKbROqtlHqk1COuuc9VFd3Iq5zVcvP/view?usp=sharing).

# Preprocessing Pipeline:

Code for preprocessing - [here](https://github.com/sanchit2843/MusicGenreClassification/blob/master/Data/data_preprocessing.py)<br>
Audio was read using librosa library at a sampling rate of 44100. The data was then converted into frequency domain using librosa.stft. The output was converted in dB scale and the spectogram was plotted and saved as image. This image was chopped into 10 parts each labelled same.

# Data Visualization
Few samples from dataset
![](https://github.com/sanchit2843/MusicGenreClassification/blob/master/assets/spectogram.png)

# Models:
We tried multiple models including densenet, efficientnet. The output layers are similar to those suggested in fast ai with concatenation pooling. Models can be seen here. The scores achieved by these models on test data are:

Click the model name to get pretrained weights. 

| Model |Accuracy|Precision|Recall|F1 Score|
|---|---|---|---|---|
|[Densenet-121](https://drive.google.com/file/d/1OH2Tc5FoKHZglc3vAW3O1CbzDFDhnW5u/view?usp=sharing)|89.85|89.90|89.85|89.84|
|[Densenet-161](https://drive.google.com/file/d/1yV10gYOnepfmSj_08g0gHzstovg7X52q/view?usp=sharing)|91.50|91.55|91.54|91.53|
|[Densenet-201](https://drive.google.com/file/d/1--I3Y-GH0xKeNHyGFKZkHE-lGkCcJBX9/view?usp=sharing)|89.27|89.28|89.30|89.28|
|[EfficientNet-b3](https://drive.google.com/file/d/1w_0S6IvG_rNvRbzyKfafdsyiA7CmG4rm/view?usp=sharing)|89.67|89.71|89.66|89.67|

Surprisingly Densenet 161 worked the best for the model. We used the pretrained weights of imagenet.  

# Training:
We used learning rate finder and cyclical learning rate. The pytorch implementation of cyclical learning rate and learning rate finder can be find here ->  [lr_finder](https://github.com/davidtvs/pytorch-lr-finder) , [cyclicallr](https://github.com/nachiket273/One_Cycle_Policy). I used a recently published optimizer named adabound. The pytorch implementation of adabound and the paper can be found here. [Paper](https://arxiv.org/abs/1902.09843), [Pytorch implementation](https://github.com/Luolc/AdaBound).
The learning rate of 1e-3 was used, the models were trained for 20 epochs. For first 10 epchs the convolution layers were freezed. For next 10 model was unfreezed. The plots are different for both phase of training. 
Training curves:

![](https://github.com/sanchit2843/MusicGenreClassification/blob/master/assets/loss_1.png)

![](https://github.com/sanchit2843/MusicGenreClassification/blob/master/assets/loss_2.png)

The sudden increase in the loss in second case is due to unfreezing of model. 

For training see exampletraining.ipynb

# Voting System

For every audio file we have chopped it into 10 spectograms with same labels to all thus significantly increasing datasize. We can use same thing at the time of prediction and getting prediction for 10 chopped spectograms and creating a voting system for the last actual prediction. This can help us to further reduce the tes accuracy. The implementation of running test on audio with voting system can be found in file exampleprediction.ipynb
