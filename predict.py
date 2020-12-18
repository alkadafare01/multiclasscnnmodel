#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
      #  model = load_model('model1.h5')
        model = tf.keras.models.load_model('model.h5')
        # summarize model
        model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][1] == 1:
            prediction = 'Pneumonia'
            return [{"image": prediction}]
            print('It is a pneumonia')
        elif result[0][2] == 1:
            prediction = 'Covid'
            return [{"image": prediction}]
            print('It is a covid')
        elif result[0][0] == 1:
            prediction = 'Normal'
            return [{"image": prediction}]
            print('It is a Normal')
            return [{ "image" : prediction}]

