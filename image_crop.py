import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os

for folders in os.listdir('./char_data'):
    for files in os.listdir('./char_data/{}'.format(folders))[0:1]:
        if(files.split('_')[5] == 'KO'):
            try:
                if not(os.path.isdir('./crop/{}/'.format(folders))):
                    os.makedirs(os.path.join('./crop/{}/'.format(folders)))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    print("Failed to create directory!!!!!")
                    raise
    for files in os.listdir('./char_data/{}'.format(folders)):
        if(files[0] != '.'):
            if(files.split('_')[5] == 'KO'):
                img = cv2.imread('./char_data/{}/'.format(folders) + files, cv2.IMREAD_GRAYSCALE)
                
                ret,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
                contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if(len(contours) != 0):
                    cnt = contours[0]
                    if(len(contours) > 1):
                        x_ = []
                        y_ = []
                        xw = []
                        yh = []
                        for l in range(len(contours)):
                            x,y,w,h = cv2.boundingRect(contours[l])
                            if(h > 5 and w > 5):
                                x_.append(x)
                                y_.append(y)
                                xw.append(x+w)
                                yh.append(y+h)

                        if(len(x_) == 0):
                            cv2.imwrite('./crop/'+ files, img)
                            break
                        else:
                            x = min(x_)
                            y = min(y_)
                            w = max(xw) - x
                            h = max(yh) - y
                    else:    
                        x,y,w,h = cv2.boundingRect(cnt)
                    a = 0
                    while(x > a and y > a and x + w > a and y + h > a):
                        a += 1 
                    #if(x < 40 or y < 40):
                    #img = img[y:y+h+20,x:x+w+20]
                    #else:
                    if(a != 0):
                        if(a > 10):
                            img = img[y-10:y+h+10,x-10:x+w+10]
                        else:
                            img = img[y-a:y+h+a,x-a:x+w+a]
                    else:
                        if(x < 10 or y < 10):
                            img = img[y:y+h,x:x+w]
                        else:
                            img = img[y-10:y+h+10,x-10:x+w+10]
                # cv2.imshow('image', img)
                # cv2.waitKey(0) 
                cv2.imwrite('./crop/{}/'.format(folders) + files, img)
                # img = cv2.flip(img, 1)
                # img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)