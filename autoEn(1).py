
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
#Encoder
path = 'dataUnCat/'

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs= {}):
    if(logs.get('loss')<0.05):
      print('reached {}% accuracy on training set and {}% accuracy on test set'.format(logs.get('acc'), logs.get('val_acc')))
      self.model.stop_training = True

callback = myCallback()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(64,64,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(4,4), padding='same'),
    #tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    #tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    #tf.keras.layers.Conv2D(64,(3, 3), padding='same', activation='relu'),
    #tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    #tf.keras.layers.Conv2D(8,(3, 3), padding='same', activation='relu'),
    #tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'),
    #tf.keras.layers.Conv2D(8,(3, 3), padding='same', activation='relu'),
    #tf.keras.layers.UpSampling2D((2, 2)),
    #tf.keras.layers.Conv2D(64,(3, 3), padding='same', activation='relu'),
    #tf.keras.layers.UpSampling2D((2, 2)),
    #tf.keras.layers.Conv2D(32,(3, 3), padding='same', activation='relu'),
    #tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2D(16,(3, 3), padding='same', activation='relu'),
    tf.keras.layers.UpSampling2D((4, 4)),
    tf.keras.layers.Conv2D(1,(3, 3), padding='same', activation='sigmoid'),
])
model.layers[0].build((64, 64, 1))


model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=300)
# Generate data from the images in a folder
batch_size = 16
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split = 0.1,
    #horizontal_flip=True,
    #vertical_flip=True,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #rotation_range=45
)
train_generator = train_datagen.flow_from_directory(
  path,
  target_size=(64, 64),
  batch_size=batch_size,
  class_mode='input',
  color_mode = 'grayscale',
  subset = 'training'
  )
validation_generator = train_datagen.flow_from_directory(
  path,
  target_size=(64, 64),
  batch_size=batch_size,
  class_mode='input',
  color_mode = 'grayscale',
  subset = 'validation',
  )
history = model.fit_generator(
          train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=10000, callbacks = [callback],
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size)
hist_df = pd.DataFrame(history.history)
hist_json_file = 'autoEnhistory2.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
model.save('autoEn2.h5')
print('done training AutoEn')
