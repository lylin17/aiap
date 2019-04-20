from wtforms import validators, SubmitField, FileField
from flask_wtf import FlaskForm
from flask import render_template,redirect,url_for
from flask import Flask
from flask import request
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import *
import os
import io
import pickle
from google.cloud import storage
import time

def load_pytorch_model():
    """Load in the pre-trained model"""   
    global model   

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, 7, padding = 0)
            self.bn1 = nn.BatchNorm2d(num_features=8)
            self.conv2 = nn.Conv2d(8, 16, 7, padding =0)
            self.bn2 = nn.BatchNorm2d(num_features=16)
            self.conv3 = nn.Conv2d(16, 32, 5, stride =2, padding =0)
            self.bn3 = nn.BatchNorm2d(num_features=32)
            self.conv4 = nn.Conv2d(32, 64, 5, stride = 2, padding =0)
            self.bn4 = nn.BatchNorm2d(num_features=64)

            self.fc1 = nn.Linear(10816,512)
            self.bn6 = nn.BatchNorm1d(num_features=512)
            self.fc2 = nn.Linear(512, 64)
            self.bn7 = nn.BatchNorm1d(num_features=64)
            self.fc3 = nn.Linear(64, 3)

            self.drop = nn.Dropout(p=0.5)
            self.drop2d = nn.Dropout2d(p=0.1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.pool2 = nn.MaxPool2d(3, 2)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = self.drop2d(F.relu(self.bn3(self.conv3(x))))
            x = self.pool1(self.drop2d(F.relu(self.bn4(self.conv4(x)))))
            x = x.view(-1, 10816) #Flatten
            x = self.drop(F.relu(self.bn6(self.fc1(x))))
            x = F.relu(self.bn7(self.fc2(x)))
            x = self.fc3(x)
            return x   

    model = Net()
    
    client = storage.Client()
    bucket = client.get_bucket('torched-fruit.appspot.com')
    blob = bucket.get_blob('selfnet_wt.pkl')
    
    model_wt = pickle.loads(blob.download_as_string())
    model.load_state_dict(model_wt)
    model.eval()

class ReusableForm(FlaskForm):
    """User entry form for entering specifics for generation"""
    # Starting seed
    file = FileField("Image File to Classify:",validators=[validators.InputRequired()])

    # Submit button
    submit = SubmitField("Classify")

def pred_fruit(model, file):

    try:        
        
        img = Image.open(file)
        width, height = img.size

        if height >= width:
            basewidth = 244
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))

        if height < width:
            hsize = 244
            hpercent = (hsize/float(img.size[1]))
            basewidth = int((float(img.size[0])*float(hpercent)))    

        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        width, height = img.size
        left = (width - 244)/2
        top = (height - 244)/2
        right = (width + 244)/2
        bottom = (height +244)/2
        img = img.crop((left, top, right, bottom))

        arr = np.array(img)

        if arr.shape == (244,244,3):
            testfruit = arr
            test = (ToTensor()(ToPILImage()(testfruit))).unsqueeze(0)
            outputs = model(test)
            _, predicted = torch.max(outputs.data, 1)
            prob = np.array(F.softmax(outputs.data,1)[0])
            text = str(int(np.max(prob)*100))+'% '+ ['Apple','Orange','Pear'][prob.argmax()]          

        else: 
            img = Image.open(errpath)
            text = 'Image dim error!'
    except:
        img = Image.open(errpath)
        text = 'Image read error!' 
    
    new_im = Image.new('RGB', (344, 360),'ivory')
    draw = ImageDraw.Draw(new_im)
    new_im.paste(img, (50,50))
    
    font = ImageFont.truetype(fontpath, size=24)
    w, h = draw.textsize(text,font)
    draw.text(((244-w)/2+50,310),text,(0,0,0),font=font) 
    
    new_im.save(filepath)

#load model outside app
load_pytorch_model()
#Flask App

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

app.secret_key = 'real super secret key'

filepath = os.path.join('static', 'predict.png')
errpath = os.path.join('static', 'error.png')
fontpath = os.path.join('static', 'arial.ttf')

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
   
    if os.path.exists(filepath): 
        os.remove(filepath)

    # Create form
    form = ReusableForm()        
    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        file = request.files['file']
        pred_fruit(model, file)
        while not os.path.exists(filepath):
            pass
        return render_template('prediction.html', output = filepath)          
    
    if request.method == 'GET':    
        return render_template('index.html', form=form)
    
@app.route('/clear')
def clear():
    return redirect(url_for('home'))

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "\nplease wait until server has fully started *"))

    app.run(host='0.0.0.0', port=5000)
