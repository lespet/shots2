#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:05:56 2019

@author: pete
"""

import face_recognition
import os
from PIL import Image
import numpy as np

import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D



# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

 
destdir = '/home/pete/face/scene1'

files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 
#model = load_model("model.h5")
model.load_weights('model.h5')

all_images=[]
predicted_classes=[]
k=0
emotion_dict = {0: "angr", 1: "disgus", 2: "fear", 3: "happy", 4: "neut", 5: "sad", 6: "surp"}
for f in files:
      print(f)     
      image = face_recognition.load_image_file(os.path.join(destdir,f))
      face_locations = face_recognition.face_locations(image)
      face_landmarks_list = face_recognition.face_landmarks(image)
      all_images.append(face_landmarks_list)
#      if k >9 :
#          break
      for face_location in face_locations:
          top, right, bottom, left = face_location
          face_image=image[top:bottom, left:right]
          f1=abs(top-bottom)+abs(right-left)
          f2=int((f1*100)/(image.shape[0]+image.shape[1]))
          pil_image = Image.fromarray(face_image)
#         pil_image.show()         
          gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
          resize = np.expand_dims(np.expand_dims(cv2.resize(gray, (48, 48)), -1), 0)
                  
          predicted_class = np.argmax(model.predict(resize))
          predicted_classes.append(predicted_class)
          pil_image.save("./cuts2/{}{}{}{}.jpg".format(f,top,emotion_dict[predicted_class],f2))
          k+=1
         
            