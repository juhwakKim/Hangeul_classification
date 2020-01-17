import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential,Model
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt


images_arr = []
labels_arr = []

for folders in os.listdir('./char_data'):
    if(len(os.listdir('./char_data/{}'.format(folders))) == 1 or len(os.listdir('./char_data/{}'.format(folders))) == 0):
        print(folders)
    for files in os.listdir('./char_data/{}'.format(folders))[0:5]:
        img = cv2.imread('./char_data/{}/'.format(folders) + files, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        images_arr.append(img)
        labels_arr.append(int(folders,16))
        
images_arr = np.reshape(images_arr,(-1,28,28,1))
images_arr = images_arr/255

labels_arr = np.array(labels_arr)
labels_arr = labels_arr.reshape(-1,1)
enc = OneHotEncoder()
enc.fit(labels_arr)
 
num_classes = 2446
batch_size = 128
epochs = 50

X_train, X_test, Y_train, Y_test = train_test_split(images_arr, labels_arr, test_size = 0.20, random_state = 42)

Y_train = enc.transform(Y_train).toarray().reshape(-1,2446)
Y_test = enc.transform(Y_test).toarray().reshape(-1,2446)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding="valid",input_shape=(28, 28, 1)))
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
          validation_data=(X_test, Y_test))


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