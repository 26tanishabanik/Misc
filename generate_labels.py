import os
import numpy as np
import pandas as pd



images = os.listdir('dataUnCat/images')
benign = os.listdir('dataCat/benign')
malignant = os.listdir('dataCat/malignant')
non_nodule = os.listdir('dataCat/non-nodule')
#print(images, benign, malignant, non_nodule)
data_dict = {'0':[], '1':[], '2':[]}
for img in images:
    if img in benign:
        data_dict['0'].append(img)
    elif img in malignant:
        data_dict['1'].append(img)
    elif img in non_nodule:
        data_dict['2'].append(img)

print(len(data_dict['0']), len(data_dict['1']), len(data_dict['2']))
label0 = pd.DataFrame(data_dict['0'])
label1 = pd.DataFrame(data_dict['1'])
label2 = pd.DataFrame(data_dict['2'])
labels = pd.concat([label0, label1, label2], axis=1)
labels_xl = 'labels.xlsx'
with open(labels_xl, mode='w') as f:
    labels.to_excel(labels_xl, index=False, header=None)
