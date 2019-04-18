# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 06:37:55 2019

@author: Govi
"""

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

base_model=VGG16(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(25,activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True



#Fit images to Model
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset3/training_set',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_set = test_datagen.flow_from_directory('dataset3/test_set',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

#fix test set, put it in the right format, watch video again
from keras.callbacks import ModelCheckpoint

filepath = "weights-VGG16-slightly-enhanced.h5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

model.fit_generator(training_set,
                         samples_per_epoch=27185,
                         epochs=100,
                         validation_data=test_set,
                         nb_val_samples=2393,
                         callbacks=[checkpoint])



#==============================================================================
# 
# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
# 
# train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)
# 
# 
# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy
# 
# step_size_train=train_generator.n//train_generator.batch_size
# model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    epochs=5)
# 
# 
#==============================================================================

"""

- add back clasess to training set for dataset 3
- start training for VGG16 slightliy enhanced
- hopw val_acc improves to 80%

- copy of dataset2, name it dataset3
- run enhance_dataset.py on it
- run pre-trained-new.py
- 

"""