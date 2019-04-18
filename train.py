# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Define Model
classifier = Sequential()

classifier.add(Convolution2D(32, (3,3), input_shape=(256, 256, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(64, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))


classifier.add(Convolution2D(64, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))
#
classifier.add(Convolution2D(128, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(128, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(output_dim = 25, activation='softmax'))

from keras import optimizers

adam = optimizers.Adam(lr=0.0001)

classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#Fit images to Model
from keras.preprocessing.image import ImageDataGenerator

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

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(256, 256),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)

#fix test set, put it in the right format, watch video again
from keras.callbacks import ModelCheckpoint

filepath = "weights-adam-0_0001.h5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

classifier.fit_generator(training_set,
                         samples_per_epoch=41754,
                         epochs=50,
                         validation_data=test_set,
                         nb_val_samples=2392,
                         callbacks=[checkpoint])

"""
To Do for classifier
3rd
 - try the classifier on the arcitecture with 8 layers. try it with the larger dataset
 
 
 1st
 -     if the model still is overfitting, try new model archetecture on regular dataset 
 -     See how well model performs with regular dataset and new model archetecture, can even
 -     add image size to 512 px, and add more layers, and epoch and check val_acc


INTERESTING IDEA
2nd
- try model with slightly enhnaced dataset with only 1 or 2 enhanced photos and see difference
in val_acc and acc, try after ideas above.
"""
