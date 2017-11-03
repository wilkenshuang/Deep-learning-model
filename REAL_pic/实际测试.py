# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:08:15 2017

@author: gd
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import cv2
import PIL.Image as Image
import pandas as pd

trimg_dir='C:/Anaconda/image/train'
teimg_dir='C:/Anaconda/image/test'

Type={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,
       '9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,
       'S':18,'T':19}
       
types=['0','1','2','3','4','5','6','7','8',
       '9','A','B','C','D','E','F','G','H',
       'S','T']

f = open('C:/Anaconda/image/训练图片.csv', encoding='utf-8')
TRAIN=pd.read_csv(f)

def preprocess(dataset):
    Label=[]
    IMG_set=np.zeros((3712,48,48))
    #从csv文件里读取参数
    for i in dataset.values:
        img_path=i[1]
        categ=Type[i[2]]
        count=i[0]
        img=Image.open(img_path)
    #建立图片集和标签集
        if img.mode!='L':
            img=img.convert('L')
        img=np.array(img.resize((48,48)),dtype=np.float32)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        img=cv2.erode(img,kernel,iterations=2)#对图片进行膨胀处理
        IMG_set[count]=img
        Label.append(categ)
    #随机展示图片集内图片
    '''
    for i in range(16):
        num=np.random.randint(0,3712)
        plt.subplot(4,4,i+1)
        img=IMG_set[num]
        label=Label[num]
        plt.imshow(img,cmap='gray')
        plt.axis('off')
        plt.title(label)
    '''
    return IMG_set,Label

Tr_img,Tr_lab=preprocess(TRAIN)#将2D图像维度降为1D
Tr_img=np.reshape(Tr_img,(-1,48*48))#-1代表自适应未指定值 
Tr_lab=np.array(Tr_lab,dtype=np.uint8)
#对标签进行多维映射   
Tr_lab= (np.arange(20) == Tr_lab[:,None]).astype(np.float32) 



#设定参数
learning_rate_base=0.0005
regularization=0.01
dropout_param=0.5
is_training=True
split_param=0.2


#划分训练集和验证集
x_train,x_valid,y_train,y_valid=train_test_split(Tr_img,Tr_lab,test_size=split_param)


def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
        
def bias_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],
                          padding='SAME')

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

x=tf.placeholder(tf.float32,shape=[None,48*48],name='x_input')
y=tf.placeholder(tf.float32,shape=[None,20],name='y_input')    
   
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
x_image=tf.reshape(x,[-1,48,48,1])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=tf.nn.relu(max_pool_2x2(h_conv1))

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=tf.nn.relu(max_pool_2x2(h_conv2))

W_fc1=weight_variable([12*12*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,12*12*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024, 20])
b_fc2 = bias_variable([20])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv))
logits=tf.nn.softmax(y_conv)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_base).minimize(cross_entropy)
correct_pred=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#saver = tf.train.Saver() # defaults to saving all variables

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for i in range(100):
        _,loss,predictions=sess.run([optimizer,cross_entropy,logits],
                                    feed_dict={x:x_train,y:y_train,keep_prob:dropout_param})
        if i%10==0:
            train_accuracy = accuracy(predictions, y_train)    
            
            print("step %d, training accuracy %g" %(i,train_accuracy))
        #optimizer.run(feed_dict={x:x_train,y:y_train,keep_prob:dropout_param})

#accuracy.eval(feed_dict={x:x_valid,y:y_valid,keep_prob:dropout_param})

