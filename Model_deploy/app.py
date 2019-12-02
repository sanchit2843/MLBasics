from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from torch import nn
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from torchvision import transforms
import matplotlib.pyplot as plt
from efficientnet_pytorch import  EfficientNet
from torchvision import models
from flask_wtf import FlaskForm
from wtforms import SubmitField, FileField
from wtforms.validators import DataRequired
import os
from flask_bootstrap import Bootstrap

# Define a flask app
app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

bootstrap = Bootstrap(app)

class Model_1(FlaskForm):

    data = FileField(validators=[DataRequired()])
    Submit = SubmitField('Run Model')

# Model saved with Keras model.save()
MODEL_PATH = 'models/classifier.h5'
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]
sm = nn.Softmax()

def prediction_bar(output,encoder):
    output = output.cpu().detach().numpy()
    a = output.argsort()
    a = a[0]

    size = len(a)
    if(size>5):
        a = np.flip(a[-5:])
    else:
        a = np.flip(a[-1*size:])
    prediction = list()
    clas = list()
    for i in a:
      prediction.append(float(output[:,i]*100))
      clas.append(str(i))
    cl = list()
    for i in a:
        cl.append(encoder[int(i)])
    plt.bar(cl,prediction)
    plt.title("Confidence score bar graph")
    plt.xlabel("Confidence score")
    plt.ylabel("Class number")
    plt.savefig('static/pred_bar.jpg')
    
#using efficientnet model based transfer learning
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet =  EfficientNet.from_name('efficientnet-b0')
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256,6)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)

classifier.load_state_dict(torch.load(MODEL_PATH,map_location=lambda storage, loc: storage))

classifier.eval()

transform = transforms.Compose([        transforms.ToPILImage(),
                                        transforms.Resize((150,150)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.43018365, 0.45747966, 0.45386454],[0.23611858, 0.23468056, 0.2432306 ])])

def im_convert(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(150,150))
    image = transform(image)
    image = image.view((1,3,150,150))
    return image


def model_predict(image_path):
    image = im_convert(image_path)
    image = image.type(torch.FloatTensor)
    pred = classifier(image)
    return sm(pred)



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

import cv2
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    
    form = Model_1()
    if form.validate_on_submit():
        f = form.data.data
        filename = f.filename
        # Get the file from post request
        # Save the file to ./uploads
        basepath = os.getcwd()
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(filename))
        f.save(file_path)
        
        # Make prediction
        pred = model_predict(file_path)
        result = np.argmax(pred.detach().numpy())
        result = encoder[int(result)]
        prediction_bar(pred,encoder)
        img = cv2.imread('')
        return redirect(url_for('output', result=result))
    return render_template('index.html', form=form)

@app.route('/output<result>', methods=['GET', 'POST'])
def output(result):
    
    return render_template('output.html', prediction=result)


if __name__ == '__main__':
#    app.run(port=5002, debug=True)

    # Serve the app with gevent
    app.run()
