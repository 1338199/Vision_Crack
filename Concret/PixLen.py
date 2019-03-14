import cv2
import os
import math

for img in os.listdir("./Save"): #read all the images in dir "Save"
    #if not hidden folder
    if not img.startswith('.'):
        #read the image img1
        img1 = cv2.imread("./Save" + "/" + img, cv2.IMREAD_GRAYSCALE)
        #get the shape
        size = img1.shape
        lengthcnt = 0;
        widthcnt = 0;
        length = [0] * size[1]
        width = [0] * size[0]
        for i in range(size[0]):
            for j in range(size[1]):
                #if this pix is in the crack-zone
                if img1[i][j] > 0 and length[j] == 0:
                    length[j] = 1
                    lengthcnt += 1
                if img1[i][j] > 0 and width[i] == 0:
                    width[i] = 1
                    widthcnt += 1

        #get the length of the crack
        totLen = math.sqrt(lengthcnt*lengthcnt+widthcnt*widthcnt)

        print(img + "中裂缝的行投影为:%d像素，占图片长度的%.2f%%，裂缝的列投影为:%d像素，占图片高度的%.2f%%。裂缝的总长约为:%.3f像素"
              % (lengthcnt,lengthcnt*100.0/size[1],widthcnt,widthcnt*100.0/size[0],totLen))
