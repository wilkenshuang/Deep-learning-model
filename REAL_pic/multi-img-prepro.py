# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:37:25 2017

@author: gd
"""

import imagenet
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt 

img_dir='C:\Anaconda\handwriting'
model_path="C:/Anaconda/image/model.ckpt"

'''
for i in os.listdir(img_dir):
    img_path=os.path.join(img_dir,i)
    img=cv2.imread(img_path)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.figure(1)
    plt.imshow(img_gray,cmap='gray')
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img=cv2.dilate(img_gray,kernel)
    img=cv2.erode(img,kernel)
    ret,bins=cv2.threshold(img,70,255,cv2.THRESH_BINARY_INV)
    plt.imshow(bins,cmap='gray')
'''
#读取图片
img_path=img_dir+'/test1.jpg'
img=cv2.imread(img_path)
#图片灰度化
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#定义结构元素
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
#对图像进行膨胀和腐蚀处理
img_e=cv2.erode(img_gray,kernel)
img_d=cv2.dilate(img_e,kernel)
#阈值处理
th=cv2.adaptiveThreshold(img_d,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,
                               7,0)#自适应阈值

ret2,th2=cv2.threshold(img_d,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#Otsu二值化

ret,th3=cv2.threshold(img_d,50,255,cv2.THRESH_BINARY_INV)#普通二值化
#th3=cv2.medianBlur(th3,3)#图像中值滤波
#轮廓检测
ret,cont1,hier1=cv2.findContours(th3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#轮廓绘制
#cv2.drawContours(img,cont1,-1,(0,255,0),2)
'''
count=0
for i in cont1:
    x, y, w, h = cv2.boundingRect(i)
    if w>=10 or h>=10:
        cv2.rectangle(img, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 2)
    
'''
#切割数字
count=0
for i in cont1:
    x, y, w, h = cv2.boundingRect(i)
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if w>=10 or h>=10:
        test=img_gray[y:y+h,x:x+w]
        
    count+=1
  

