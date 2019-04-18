# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:35:36 2019

@author: Govi
"""

import csv
import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model

classifier = load_model("weights-vgg.h5")

dataset = pd.read_csv('dataset3/test_images.tsv', sep='\t')

files_names = dataset['file']
dataset['label'] = ''
dataset.to_csv('dataset3/test_pred.tsv', sep='\t')

dataset2 = pd.read_csv('dataset3/test_pred.tsv', sep='\t')


source = os.getcwd()



classes = {'Asparagus, string beans & brussels sprouts': 0,
             'Bananas, apples & pears': 1,
             'Bell peppers, zucchinis & eggplants': 2,
             'Berries & cherries': 3,
             'Broccoli, cauliflowers, carrots & radish': 4,
             'Cheese': 5,
             'Citrus fruits': 6,
             'Cucumber, tomatoes & avocados': 7,
             'Eggs': 8,
             'Fish': 9,
             'Fresh bread': 10,
             'Fresh herbs': 11,
             'Kiwis, grapes & mango': 12,
             'Lunch & Deli Meats': 13,
             'Milk': 14,
             'Minced meat & meatballs': 15,
             'Nectarines, peaches & apricots': 16,
             'Onions, leek, garlic & beets': 17,
             'Pineapples, melons & passion fruit': 18,
             'Pork, beef & lamb': 19,
             'Potatoes': 20,
             'Poultry': 21,
             'Pre-baked breads': 22,
             'Pudding, yogurt & quark': 23,
             'Salad & cress': 24}
counter = 0
for file in files_names:
    test_image = image.load_img(source + '/dataset3/validation_set/' + file, target_size = (224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    
    pred = list(classes.keys())[list(classes.values()).index(np.argmax(result))]
    
    dataset2.iloc[counter:counter+1, 2:3] = pred
    counter += 1




dataset2.to_csv('dataset3/submission.tsv', sep='\t')
