import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import itertools, os, time
import numpy as np
from Model import get_Model
from parameter import letters
import argparse
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

rc('font', family="NanumGothic")


K.set_learning_phase(0)


def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr



label_name_list = np.load('./numpy/label_name_list.npy',allow_pickle = True)

# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights('LSTM+BN5--18--1.635.hdf5')
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


test_imgs = os.listdir('./DB')
total = 0
acc = 0
letter_total = 0
letter_acc = 0
start = time.time()

j = 1
fig = plt.figure(figsize=(15, 6)) 

for test_img in test_imgs:
    img = cv2.imread('./DB/' + test_img, cv2.IMREAD_GRAYSCALE)

    # img = img.astype(np.float32)
    img = cv2.resize(img, (128, 64))
    img_pred = (img / 255.0)
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)

    net_out_value = model.predict(img_pred)

    out_best = list(np.argmax(net_out_value[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    name_pred = ""
    for i in out_best:
        name_pred += chr(int(label_name_list[i-1],16))
    print(name_pred)
    if(j> 20):
        break

    plt.subplot(5, 4, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(img, cmap='gray',aspect='auto')  # cmap='gray' is for black and white picture.
    plt.title('predict = {},{}'.format(name_pred,test_img[:3]))  
    plt.axis('off')  # do not show axis value
    plt.tight_layout()   # automatic padding between subplots
plt.savefig('cas.png')
plt.show()
    # pred_texts = decode_label(net_out_value)

    # for i in range(min(len(pred_texts), len(test_img[0:-4]))):
    #     if pred_texts[i] == test_img[i]:
    #         letter_acc += 1
    # letter_total += max(len(pred_texts), len(test_img[0:-4]))

    # if pred_texts == test_img[0:-4]:
    #     acc += 1
    # total += 1
    # print('Predicted: %s  /  True: %s' % (label_to_hangul(pred_texts), label_to_hangul(test_img[0:-4])))
    
    # cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
    # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

    #cv2.imshow("q", img)
    #if cv2.waitKey(0) == 27:
    #   break
    #cv2.destroyAllWindows()

# end = time.time()
# total_time = (end - start)
# print("Time : ",total_time / total)
# print("ACC : ", acc / total)
# print("letter ACC : ", letter_acc / letter_total)
