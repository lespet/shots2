#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:16:42 2019

@author: pete
"""

#import face_recognition
import os
#from PIL import Image
import numpy as np
from keras.models import load_model     
destdir = '/home/pete/face/scene1'
model = load_model('model2.hdf5')