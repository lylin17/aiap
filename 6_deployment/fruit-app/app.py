from wtforms import validators, SubmitField, FileField
from flask_wtf import FlaskForm
from flask import render_template,redirect,url_for
from flask import Flask
from flask import request
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import tensorflow as tf
import os

def load_keras_model():
    """Load in the pre-trained model"""   
    global model
    model = load_model('vgg16.h5')
            
    global graph
    graph = tf.get_default_graph()

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
            with graph.as_default():
                testfruit = arr
                testfruit = np.expand_dims(testfruit,axis=0)
                testfruit = preprocess_input(testfruit)
                pred  = model.predict(testfruit)
                text = str(int(np.max(pred)*100))+'% '+ ['Apple','Orange','Pear'][pred.argmax()]          

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
load_keras_model()
#Flask App

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"

app.secret_key = 'super secret key'

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
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "\nplease wait until server has fully started *"))

    app.run(host='0.0.0.0', port=5000)
