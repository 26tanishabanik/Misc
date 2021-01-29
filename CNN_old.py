import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs= {}):
    if(logs.get('acc')>0.98):
      print('reached {}% accuracy on training set and {}% accuracy on test set'.format(logs.get('acc'), logs.get('val_acc')))
      self.model.stop_training = True

callback = myCallback()

#CNN
batch_size = 16
train_image_generator = ImageDataGenerator(
  rescale = 1./255,
  validation_split = 0.1
)
train_data = train_image_generator.flow_from_directory(batch_size = batch_size,
                                                       directory = 'dataCat',
                                                       shuffle = True,
                                                       target_size = (128, 128),
                                                       color_mode = 'grayscale',
                                                       class_mode = 'categorical',
                                                       subset = 'training',
                                                       seed = 2)
validation_data = train_image_generator.flow_from_directory(batch_size = batch_size,
                                                       directory = 'dataCat',
                                                       shuffle = True,
                                                       target_size = (128, 128),
                                                       color_mode = 'grayscale',
                                                       class_mode = 'categorical',
                                                       subset = 'validation',
                                                       seed = 2)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3,3),activation = 'relu', input_shape = [128,128 , 1]),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['acc', 'Precision', 'Recall',tf.keras.metrics.SpecificityAtSensitivity(0.98), tf.keras.metrics.SensitivityAtSpecificity(0.98), 'TruePositives', 'TrueNegatives', 'AUC']
)
history = model.fit(train_data, steps_per_epoch = train_data.samples//batch_size, validation_data = validation_data, validation_steps = validation_data.samples//batch_size, epochs = 10000, callbacks = [callback])
hist_df = pd.DataFrame(history.history)
hist_json_file = 'CNNhistory.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
model.save('CNN.h5')
print('done training CNN')
