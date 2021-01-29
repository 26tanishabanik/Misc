import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd



model = tf.keras.models.load_model('NN2.h5')
data_gen = ImageDataGenerator(rescale=1./255)
test_data = data_gen.flow_from_directory(
    directory = 'dataUnCat',
    target_size = (128, 128),
    color_mode = 'grayscale',
    class_mode = 'categorical'
)



CNN1 = tf.keras.models.load_model('CNN.h5')
encoder1 = tf.keras.Model(CNN1.input, CNN1.layers[-7].output)
prediction_en1 = np.array(encoder1.predict(test_data), dtype=np.float)
CNN2 = tf.keras.models.load_model('CNN2.h5')
encoder2 = tf.keras.Model(CNN2.input, CNN2.layers[-7].output)
prediction_en2 = np.array(encoder2.predict(test_data), dtype=np.float)
input_nn = np.concatenate((prediction_en1, prediction_en2), axis=1).reshape(301,30,30, 32)



labels = []
label_dict = pd.read_excel('labels.xlsx')
label_dict = {'0': label_dict[0].tolist(), '1': label_dict[1].tolist(), '2': label_dict[2].tolist()}
for name in test_data.filepaths:
    name = name[18:] if 'opy' not in name[18:] else name[25:]
    name = '(' + name if '((' not in name else name
    #print(name)
    if name in label_dict['0']:
        labels.append([1, 0, 0])
    elif name in label_dict['1']:
        labels.append([0, 1, 0])
    elif name in label_dict['2']:
        labels.append([0, 0, 1])
    else:
        print('notFound {}'.format(name))
labels = np.array(labels).reshape(301, 3)





model.evaluate(input_nn, labels)
