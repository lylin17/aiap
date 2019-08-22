# Flask App for Classifying Fruits

[![Fruit App][shell_img]][shell_link]

[shell_img]: https://dl.dropboxusercontent.com/s/d5w4yynxzqar9q5/apple-pear-orange.jpg
[shell_link]: https://peaora-app.herokuapp.com/

This project shows how to use [Flask](http://flask.pocoo.org/) to deploy a CNN-based fruit classifier on Heroku. The CNN-model was implemented under the keras framework based on transfer learning from a VGG16 model trained on Imagenet. The app allows users to upload images to the server and returns a prediction of whether the image contains apples, oranges or pears with the associated probability. Files uploaed must be images in a format that is readable by PILLOW (eg. png, jpg, bmp). If an unreadable image or an image with dimension !=3 is uploaded, image read error and image dimension error will be shown respectively. The user can upload a new image using the link at the top left corner. [Click on the fruit image above to launch app]

Before running or deploying this application, install the dependencies using
[pip](http://pip.readthedocs.io/en/stable/):

    pip install -r requirements.txt
    
Run this application using:

    python app.py
