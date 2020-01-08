#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:00:52 2019

@author: pete
"""

import face_recognition
image = face_recognition.load_image_file("a3.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)
face_locations = face_recognition.face_locations(image)