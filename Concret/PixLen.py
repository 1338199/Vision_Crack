import cv2
import os

for img in os.listdir("./Save"): #read all the images in dir "Save"
    #if not hidden folder
    if not img.startswith('.'):
        #read the image img1
        img1 = cv2.imread("./Save" + "/" + img, cv2.IMREAD_GRAYSCALE)
        #get the shape
        size = img1.shape
        lengthcnt = 0;
        length = [0] * size[1]
        for i in range(size[0]):
            for j in range(size[1]):
                #if this pix is in the crack-zone
                if img1[i][j] > 0 and length[j] == 0:
                    length[j] = 1
                    lengthcnt += 1

        print(img + "中裂缝的像素长为:%d，占图片长度的%.2f%%" % (lengthcnt,lengthcnt*100.0/size[1]))
