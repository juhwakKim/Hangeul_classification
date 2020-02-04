import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from matplotlib import rc, font_manager

import cv2
# from resnet import resnet_build
import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential,Model
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt

rc('font', family="NanumGothic")

lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.1, patience=20,min_lr=0.0001)
MODEL_SAVE_FOLDER_PATH = './model/cnn_alpha'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                verbose=1, save_best_only=True)

images_arr = []
labels_arr = []

for folders in os.listdir('./char_data'):
    if(len(os.listdir('./char_data/{}'.format(folders))) == 1 or len(os.listdir('./char_data/{}'.format(folders))) == 0):
        print(folders)
    for files in os.listdir('./char_data/{}'.format(folders)):
      if(files[0] != '.'):
        img = cv2.imread('./char_data/{}/'.format(folders) + files, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_AREA)/255
        images_arr.append(img)
        labels_arr.append(int(folders,16))
print(len(os.listdir('./char_data')))
images_arr = np.reshape(images_arr,(-1,64,64,1))
# images_arr = images_arr/255

labels_arr = np.array(labels_arr)
labels_arr_ = np.unique(labels_arr).reshape(-1,1)
labels_arr = labels_arr.reshape(-1,1)

enc = OneHotEncoder()
enc.fit(labels_arr_)
 
num_classes = 2447
batch_size = 128
epochs = 50

X_train, X_test, Y_train, Y_test = train_test_split(images_arr, labels_arr, test_size = 0.20, random_state = 42)

Y_train = enc.transform(Y_train).toarray().reshape(-1,2447)
Y_test = enc.transform(Y_test).toarray().reshape(-1,2447)

# model = resnet_build.build(64,num_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding="valid",input_shape=(64, 64, 1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.45))

model.add(Conv2D(64, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Conv2D(128, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Conv2D(256, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.45))

model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001,epsilon=1e-04),
              metrics=['accuracy'])

model.summary()

hist = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[cb_checkpoint,lr_reduce])


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(X_test, Y_test, verbose=0)
pred = model.predict(X_test)
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(pred,1)
print(np.argmax(pred,axis=1))
print(score[1]*100)
#print("복원된 모델의 정확도: {:5.2f}%".format(100*score))
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    list_.append(i)

ROW = 5
COLUMN = 4
f = plt.figure(figsize=(5,4))
f.set_figheight(10)
f.set_figwidth(10)
j = 1
for i in list_[0:20]:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test[i].reshape(64, 64)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('predict = {},{}'.format(chr(labels_arr_[np.argmax(pred[i],0)]),chr(enc.inverse_transform(Y_test[i].reshape(1,-1))[0][0])),fontsize=15)  
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('cnn_{}.png'.format(score[1]*100))
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn_acc.png')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn_loss.png')
plt.show()