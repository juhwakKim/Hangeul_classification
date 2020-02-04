import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os, random
import numpy as np
from parameter import letters
import random

# # Input data generator
def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: letters.index(x), text))


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=3):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        self.a = 0
        self.img_name_list_train = []
        self.img_label_list_train = []
        self.img_name_list_val = []
        self.img_label_list_val = []
        self.label_list = []
        self.label_name_list = []

    ## samples의 이미지 목록들을 opencv로 읽어 저장하기, texts에는 label 저장
    def build_data(self):


        # for i, folders in enumerate(os.listdir('./crop')):
        #     img_name_list_train_ = []
        #     img_label_list_train_ = []
        #     img_name_list_val_ = []
        #     img_label_list_val_ = []

        #     self.label_list.append(i)
        #     print(i)
        #     self.label_name_list.append(folders)
        #     for j,files in enumerate(os.listdir('./crop/{}'.format(folders))):
        #         if(j < int(len(os.listdir('./crop/{}'.format(folders))) * 0.8)):
        #             img_name_list_train_.append('./crop/{}/'.format(folders) + files)
        #             img_label_list_train_.append(i)
        #         else:
        #             img_name_list_val_.append('./crop/{}/'.format(folders) + files)
        #             img_label_list_val_.append(i)
        #     self.img_name_list_train.append(img_name_list_train_)
        #     self.img_label_list_train.append(img_label_list_train_)
        #     self.img_name_list_val.append(img_name_list_val_)
        #     self.img_label_list_val.append(img_label_list_val_)

        # self.img_name_list_train = np.array(self.img_name_list_train)
        # self.img_label_list_train = np.array(self.img_label_list_train)
        # self.img_name_list_val = np.array(self.img_name_list_val)
        # self.img_label_list_val = np.array(self.img_label_list_val)
        # self.label_list = np.array(self.label_list)
        # self.label_name_list = np.array(self.label_name_list)
        self.img_name_list_train = np.load('./numpy/img_name_list_train.npy',allow_pickle = True)
        self.img_label_list_train = np.load('./numpy/img_label_list_train.npy',allow_pickle = True)
        self.img_name_list_val = np.load('./numpy/img_name_list_val.npy',allow_pickle = True)
        self.img_label_list_val = np.load('./numpy/img_label_list_val.npy',allow_pickle = True)
        self.label_list = np.load('./numpy/label_list.npy',allow_pickle = True)
        self.label_name_list = np.load('./numpy/label_name_list.npy',allow_pickle = True)


        print(self.n, " Image Loading start...")


    def next_sample(self):      ## index max -> 0 으로 만들기
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 4)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                random_idx = np.random.choice(len(self.img_name_list_train),3) 
                rand_1 = random.randint(0, len(self.img_name_list_train[random_idx[0]]) - 1)
                rand_2 = random.randint(0, len(self.img_name_list_train[random_idx[1]]) - 1)
                rand_3 = random.randint(0, len(self.img_name_list_train[random_idx[2]]) - 1)
                
                img_1 = cv2.imread(self.img_name_list_train[random_idx[0]][rand_1], cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(self.img_name_list_train[random_idx[1]][rand_2], cv2.IMREAD_GRAYSCALE)
                img_3 = cv2.imread(self.img_name_list_train[random_idx[2]][rand_3], cv2.IMREAD_GRAYSCALE)
                w_add = img_1.shape[1]+img_2.shape[1]+img_3.shape[1]
                h_max = max([img_1.shape[0],img_2.shape[0],img_3.shape[0]])
                img = np.ones((h_max,w_add),dtype=np.uint8)*255
                w_ = 0
                for img_ in [img_1,img_2,img_3]:
                    img_gap = int((h_max - img_.shape[0])/2)
                    img[img_gap:img_.shape[0]+img_gap,w_:w_+img_.shape[1]] = img_
                    w_ += img_.shape[1]
                # cv2.imshow("asd",img)
                # print(chr(int(self.label_name_list[self.img_label_list_train[random_idx[0]][rand_1]],16)))
                # print(chr(int(self.label_name_list[self.img_label_list_train[random_idx[1]][rand_2]],16)))
                # print(chr(int(self.label_name_list[self.img_label_list_train[random_idx[2]][rand_3]],16)))
                # cv2.waitKey(0)
                img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32)
                img = img.reshape(((128,64,1)))
                
                X_data[i] = img/255
                
                Y_data[i,:] = np.array([self.img_label_list_train[random_idx[0]][rand_1],self.img_label_list_train[random_idx[1]][rand_2],self.img_label_list_train[random_idx[2]][rand_3]])
                # Y_data[i,:] = np.array([self.img_label_list_train[random_idx[0]][rand_1]])
                label_length[i] = 3

            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0
            # print("X_data",X_data)
            # print("Y_data",Y_data)
            # print("input_length",input_length)
            # print("label_length",label_length)
            yield (inputs, outputs)

    def next_batch_val(self):       ## batch size만큼 가져오기
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 4)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                random_idx = np.random.choice(len(self.img_name_list_val),3) 
                rand_1 = random.randint(0, len(self.img_name_list_val[random_idx[0]]) - 1)
                rand_2 = random.randint(0, len(self.img_name_list_val[random_idx[1]]) - 1)
                rand_3 = random.randint(0, len(self.img_name_list_val[random_idx[2]]) - 1)
                
                img_1 = cv2.imread(self.img_name_list_val[random_idx[0]][rand_1], cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(self.img_name_list_val[random_idx[1]][rand_2], cv2.IMREAD_GRAYSCALE)
                img_3 = cv2.imread(self.img_name_list_val[random_idx[2]][rand_3], cv2.IMREAD_GRAYSCALE)
                w_add = img_1.shape[1]+img_2.shape[1]+img_3.shape[1]
                h_max = max([img_1.shape[0],img_2.shape[0],img_3.shape[0]])
                img = np.ones((h_max,w_add),dtype=np.uint8)*255
                w_ = 0
                for img_ in [img_1,img_2,img_3]:
                    img_gap = int((h_max - img_.shape[0])/2)
                    img[img_gap:img_.shape[0]+img_gap,w_:w_+img_.shape[1]] = img_
                    w_ += img_.shape[1]
                img = cv2.resize(img,(64,128),interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32)
                img = img.reshape(((128,64,1)))
                X_data[i] = img/255

                Y_data[i,:] = np.array([self.img_label_list_val[random_idx[0]][rand_1],self.img_label_list_val[random_idx[1]][rand_2],self.img_label_list_val[random_idx[2]][rand_3]])
                # Y_data[i,:] = np.array([s/elf.img_label_list_val[random_idx[0]][rand_1]])

                label_length[i] = 3

            # dict 형태로 복사
            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1) -> 모든 원소 0

            yield (inputs, outputs)