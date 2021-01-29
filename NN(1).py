import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs= {}):
    if(logs.get('acc')>0.99):
      print('reached {}% accuracy on training set and {}% accuracy on test set'.format(logs.get('acc'), logs.get('val_acc')))
      self.model.stop_training = True

callback = myCallback()




#encoder_arc = tf.keras.models.load_model('autoEn.h5')
#encoder = keras.Input(encoder_arc[0])
#enc_op = keras.layers.Flatten()(encoder_arc[-1])
#encoder = tf.keras.Model(encoder_arc.input, encoder_arc.layers[-6].output)
batch_size = 16
path = 'dataUnCat'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
  path,
  target_size=(128, 128),
  batch_size=batch_size,
  class_mode='input',
  color_mode = 'grayscale',
)
'''train_generator_CNN = train_datagen.flow_from_directory(
path,
target_size=(128, 128),
batch_size=batch_size,
class_mode='input',
color_mode = 'grayscale',
)'''
#print(train_generator.filepaths)


encoderDims = 196
CNN1 = tf.keras.models.load_model('CNN.h5')
encoder1 = tf.keras.Model(CNN1.input, CNN1.layers[-7].output)
prediction_en1 = np.array(encoder1.predict(train_generator), dtype=np.float)
#plt.imshow(prediction_en[0, :, :, 0], cmap='gray')
print(prediction_en1.shape)
#en_arr = np.array(prediction_en, dtype=np.float)[:, :, :,  0].reshape(301, encoderDims)
#en_arr = en_arr/np.max(en_arr)
'''prediction_cnn = CNN.predict(train_generator)
cnn_arr = np.array(prediction_cnn, dtype=np.float).reshape(301, 3)
input_nn = np.concatenate((en_arr, cnn_arr), axis=1).reshape(301,6275,1)'''
CNN2 = tf.keras.models.load_model('CNN2.h5')
encoder2 = tf.keras.Model(CNN2.input, CNN2.layers[-7].output)
prediction_en2 = np.array(encoder2.predict(train_generator), dtype=np.float)
input_nn = np.concatenate((prediction_en1, prediction_en2), axis=1).reshape(301,30,30, 32)
input_nn = input_nn/np.max(input_nn)
print(input_nn.shape)
#input_nn = en_arr.reshape(301, encoderDims, 1)

trainSplit = 0.9

#train_split = input_nn[:int(len(input_nn)*trainSplit)]
#valid_split = train_split = input_nn[int(len(input_nn)*trainSplit):]
labels = []
label_dict = pd.read_excel('labels.xlsx')
label_dict = {'0': label_dict[0].tolist(), '1': label_dict[1].tolist(), '2': label_dict[2].tolist()}
#print(label_dict)
for name in train_generator.filepaths:
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
#print(labels)
labels = np.array(labels).reshape(301, 3)
#train_labels = labels[:int(len(input_nn)*trainSplit)]
#valid_labels = labels[int(len(input_nn)*trainSplit):]

#print(en_arr)
#plt.show()
#print(prediction_cnn)



model = tf.keras.models.Sequential([
    #tf.keras.layers.Dense(16387, activation = 'relu'),
    #tf.keras.layers.Dense(4096, activation = 'relu'),
    #tf.keras.layers.Dense(2048, activation = 'relu'),
    #tf.keras.layers.Dense(1024, activation = 'relu'),
    #tf.keras.layers.Conv1D(8, 3, input_shape = (encoderDims, 1)),
    #tf.keras.layers.MaxPooling1D(3),
    #tf.keras.layers.Conv2D(16, (3, 3), input_shape = [30, 30, 32]),
    #tf.keras.layers.MaxPooling2D((2,2)),
    #tf.keras.layers.Conv2D(32, (3, 3)),
    #tf.keras.layers.Conv2D(64, (3, 3)),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(4096, activation = 'relu'),
    #tf.keras.layers.Dropout(0.9),
    #tf.keras.layers.Dense(2048, activation = 'relu'),
    #tf.keras.layers.Dropout(0.9),
    #tf.keras.layers.Dense(1024, activation = 'relu'),
    #tf.keras.layers.Dropout(0.9),
    #tf.keras.layers.Dense(4096, activation = 'relu'),
    #tf.keras.layers.Dropout(0.9),
    tf.keras.layers.Dense(256, activation = 'relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc', 'Precision', 'Recall',tf.keras.metrics.SpecificityAtSensitivity(0.98), tf.keras.metrics.SensitivityAtSpecificity(0.98), 'TruePositives', 'TrueNegatives', 'AUC'])
history = model.fit(x = input_nn, y = labels, batch_size = 6, validation_split = 1 - trainSplit, validation_batch_size = 10, epochs = 10000, callbacks = [callback], shuffle=True)
hist_df = pd.DataFrame(history.history)
hist_json_file = 'NNhistory2.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
model.save('NN2.h5')
print('done training DNN')
