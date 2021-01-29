from tensorflow.keras.models import load_model
from collections import deque
import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

model = load_model(r"C:\Users\KIIT\Downloads\my_model.h5")
lb = pickle.loads(open(r"C:\Users\KIIT\Downloads\lb.pickle", "rb").read())
# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)
vs = cv2.VideoCapture(r"C:\Users\KIIT\Downloads\video(1).mp4")
writer = None
(W, H) = (None, None)
# loop over frames from the video file stream
while True:
  (grabbed, frame) = vs.read()
  if not grabbed:
    break
  if W is None or H is None:
    (H, W) = frame.shape[:2]
  output = frame.copy()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame = cv2.resize(frame, (224, 224)).astype("float32")
  frame -= mean
  preds = model.predict(np.expand_dims(frame, axis=0))[0]
  Q.append(preds)
  results = np.array(Q).mean(axis=0)
  i = np.argmax(results)
  label = lb.classes_[i]
  text = "activity: {}".format(label)
  cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
  if writer is None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(r"C:\Users\KIIT\Downloads\football.avi", fourcc, 30,(W, H), True)
  writer.write(output)
  cv2.imshow("Output", output)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break
writer.release()
vs.release()

