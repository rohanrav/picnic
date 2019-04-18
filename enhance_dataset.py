#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:57:54 2019

@author: rohi
"""

import cv2
import os
import numpy as np
import imutils
    
def main():
    basepath = os.getcwd() + "/dataset3/training_set"
    for test_class in os.listdir(basepath):
        for entry in os.listdir(basepath + "/" + test_class):
            image = cv2.imread(basepath + "/" + test_class + "/" + entry)
            
#==============================================================================
#             rotate15 = imutils.rotate(image.copy(), 15)
#             cv2.imwrite(basepath + "/" + test_class + "/" + "rotate15-" + entry,rotate15)
#             
#==============================================================================
            rotate105 = imutils.rotate(image.copy(), 105)
            rotate105 = noisy("speckle", rotate105)
            cv2.imwrite(basepath + "/" + test_class + "/" + "rotate105_speckle-" + entry,rotate105)
            
            flip = cv2.flip(image.copy(), 0)
            cv2.imwrite(basepath + "/" + test_class + "/" + "flip-" + entry,flip)
#==============================================================================
#              
#             rotate90 = imutils.rotate(image.copy(), 90)
#             cv2.imwrite(basepath + "/" + test_class + "/" + "rotate_90-" + entry,rotate90)
#              
#             rotate180 = imutils.rotate(image.copy(), 180)
#             cv2.imwrite(basepath + "/" + test_class + "/" + "rotate_180-" + entry,rotate180)
#              
#==============================================================================
#==============================================================================
#             wn = noisy("s&p", image.copy())
#             cv2.imwrite(basepath + "/" + test_class + "/" + "white_noise-" + entry,wn)
#             
#==============================================================================
            speckle = noisy("speckle", image.copy())
            cv2.imwrite(basepath + "/" + test_class + "/" + "speckle-" + entry,speckle)

def noisy(noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.07
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy 