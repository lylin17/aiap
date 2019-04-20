# Flask App for Classifying Fruits

[![Torched Fruit][shell_img]][shell_link]

[shell_img]: https://dl.dropboxusercontent.com/s/d5w4yynxzqar9q5/apple-pear-orange.jpg
[shell_link]: https://torched-fruit.appspot.com/

This project shows how to use [Flask](http://flask.pocoo.org/) to deploy a CNN-based fruit classifier on Google Cloud Platform App Engine Flexible. The CNN-model was implemented under the pytorch framework based on Cyclical learning rate with Restarts and trained from scratch. The app allows users to upload images to the server and returns a prediction of whether the image contains apples, oranges or pears with the associated probability. Files uploaed must be images in a format that is readable by PILLOW (eg. png, jpg, bmp). If an unreadable image or an image with dimension !=3 is uploaded, image read error and image dimension error will be shown respectively. The user can upload a new image using the link at the top left corner. [Click on the fruit image above to launch app]

Before running or deploying this application, install the dependencies using
[pip](http://pip.readthedocs.io/en/stable/):

    pip install -r requirements.txt
    
To launch this application locally, download the pytorch state_dict from https://www.dropbox.com/s/eosp9poooadvsv0/selfnet_wt.pkl , modify the code in main.py and run:

    python main.py


