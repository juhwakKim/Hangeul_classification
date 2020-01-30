from keras import backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *
from Model import get_Model

K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('LSTM+BN5--27--23.855.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = './DB/train/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

# valid_file_path = './DB/test/'
# tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
# tiger_val.build_data()

ada = tf.keras.optimizers.Adadelta()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)
model.summary()

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=100,#int(tiger_train.n / batch_size),
                    epochs=60,
                    callbacks=[checkpoint],
                    validation_data=tiger_train.next_batch_val(),
                    validation_steps=20,)#int(tiger_train.n / val_batch_size))
