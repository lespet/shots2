#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:51:14 2019

@author: pete
"""
import face_recognition
import os
from PIL import Image
import numpy as np
#from keras.models import load_model     
destdir = '/home/pete/face/scene1'

files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 
#model = load_model("model1.hdf5")
all_images=[]
predicted_classes=[]
for f in files:
      print(f)     
      image = face_recognition.load_image_file(os.path.join(destdir,f))
      face_locations = face_recognition.face_locations(image)
      face_landmarks_list = face_recognition.face_landmarks(image)
      all_images.append(face_landmarks_list)
      for face_location in face_locations:
          top, right, bottom, left = face_location
          face_image=image[top:bottom, left:right]
          pil_image = Image.fromarray(face_image)
#          pil_image.show()         
#          predicted_class = np.argmax(model.predict(face_image))
#         predicted_classes.append(predicted_class)
          pil_image.save("./cuts/{}{}.jpg".format(f,top))