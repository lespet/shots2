#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:51:14 2019

@author: pete
"""
import face_recognition
import os


files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
      print(f)
      
destdir = '/home/pete/face/scene1'

files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 

all_images=[]
for f in files:
      print(f)     
      print(os.path.join(destdir,f)) 
      image = face_recognition.load_image_file(os.path.join(destdir,f))
      face_landmarks_list = face_recognition.face_landmarks(image)
      all_images.append(face_landmarks_list)
      