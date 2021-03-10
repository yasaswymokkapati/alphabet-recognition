##Import modules##

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

##setting an HTTPS context to fetch the data from openml##

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

##fetch the data##

X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

##giving the classes##

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

##splitting the data and scaling it##

X_train , X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 7500, test_size = 2500)

X_train_scale = X_train / 255.0
X_test_scale = X_test / 255.0

##Logistic Regression##

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scale, y_train)
y_pred = clf.predict(X_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

##Starting the camera##
cap = cv2.VideoCapture(0)
while(True):
    ##capture frame by frame##
    try:
        ret, frame = cap.read()
        ##operating frames##
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ##focusing frame##
        height, width = gray.shape
        upper_left = (int(width/2 - 60), int(height/2 - 60))
        lower_right = (int(width/2 + 60), int(height/2 + 60))
        cv2.rectangle(gray, upper_left, lower_right, (0, 255, 0), 2)
        ##converting the focus area into ROI##
        ROI = gray[upper_left[1] : lower_right[1], upper_left[0] : lower_right[0]]
        ##converting cv2 image into PIL format##
        im_pil = Image.fromarray(ROI)
        image_focus = im_pil.convert('L')
        image_focus_resized = image_focus.resize((28, 28), Image.ANTIALIAS)
        ##inverting the inverted image##
        image_focus_resized_inverted = PIL.ImageOps.invert(image_focus_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_focus_resized_inverted, pixel_filter)
        ## clipping image##
        image_focus_resized_inverted_scale = np.clip(image_focus_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_focus_resized_inverted)
        ##changing data into array##
        image_focus_resized_inverted_scale = np.asarray(image_focus_resized_inverted_scale)/max_pixel
        ##create a test sample and make a prediction##
        test_sample = np.array(image_focus_resized_inverted_scale).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print('Predicted class is ', test_pred)
        ##key control to control the camera##
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()