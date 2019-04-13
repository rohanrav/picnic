# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import shutil

dataset = pd.read_csv('dataset/train_labels.tsv', sep='\t')

files_names = dataset['file']
label_names = dataset['label']

folders_to_be_created = np.unique(list(label_names))

source = os.getcwd()


for new_path in folders_to_be_created:
    if not os.path.exists(str(source + "/training_set" + "/" + new_path)):
         os.makedirs(str(source + "/training_set" + "/" + new_path))
         
folders = folders_to_be_created.copy()

print(folders)

for i in range(len(files_names)):
    
    current_img = files_names[i]
    current_label = label_names[i]
    
    shutil.move(("dataset/train/" + str(current_img)), (source + "/training_set" + "/" + current_label))
       


