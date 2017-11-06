# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:08:14 2017

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
#定义参数
IMAGE_WIDTH,IMAGE_HEIGHT=48,48
lr_rate=0.005
split_param=0.2
#对标签进行多维映射   
Tr_lab= (np.arange(20) == Tr_lab[:,None]).astype(np.float32) 

#划分训练集和验证集
x_train,x_valid,y_train,y_valid=train_test_split(Tr_img,Tr_lab,test_size=split_param)

def get_batch(imgset,labset,batch_size=200):
    batch_x=np.zeros([batch_size,IMAGE_WIDTH*IMAGE_HEIGHT])
    batch_y=np.zeros([batch_size,20])
    for i in range(batch_size):
        num=np.random.randint(1,imgset.shape[0])
        batch_x[i]=imgset[num]
        batch_y[i]=labset[num]
    return batch_x,batch_y      

#建立CNN模型
sess=tf.InteractiveSession()
X=tf.placeholder(tf.float32,[None,IMAGE_WIDTH*IMAGE_HEIGHT])
Y=tf.placeholder(tf.float32,[None,20])
keep_prob=tf.placeholder(tf.float32)
lr_base=tf.placeholder(tf.float32)
train_phase=tf.placeholder(tf.bool)
#定义Batch Normalization
def batch_norm_layer(x,train_phase,scope_bn):
    with tf.variable_scope(scope_bn):
        beta=tf.Variable(tf.zeros(shape=[x.shape[-1]]),name='beta',trainable=True)
        gamma=tf.Variable(tf.constant(1.0,shape=[x.shape[-1]]),name='gamma',trainable=True)
        #axies=np.arange(len(x.shape)-1)
        batch_mean,batch_var=tf.nn.moments(x,[0,1,2],name='moments')
        ema=tf.train.ExponentialMovingAverage(decay=0.5)
        
        def mean_var_with_update():
            ema_apply_op=ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean),tf.identity(batch_var)
                
        mean,var=tf.cond(train_phase,mean_var_with_update,
                         lambda:(ema.average(batch_mean),ema.average(batch_var)))
        normed=tf.nn.batch_normalization(x,mean,var,beta,gamma,1e-3)    
    return normed
    
#定义CNN
def handmade_CNN(w_alpha=0.01,b_alpha=0.05):
    x=tf.reshape(X,shape=[-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    
    #4层卷积层
    w_c1=tf.Variable(w_alpha*tf.random_normal([5,5,1,32]))
    b_c1=tf.Variable(w_alpha*tf.random_normal([32]))
    conv1=tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1)
    conv1=batch_norm_layer(conv1,train_phase,scope_bn='bn_1')
    conv1=tf.nn.relu(conv1)
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1=tf.nn.dropout(conv1,keep_prob=keep_prob)
    
    w_c2=tf.Variable(w_alpha*tf.random_normal([5,5,32,64]))
    b_c2=tf.Variable(w_alpha*tf.random_normal([64]))
    conv2=tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2)
    conv2=batch_norm_layer(conv2,train_phase,scope_bn='bn_2')
    conv2=tf.nn.relu(conv2)
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2=tf.nn.dropout(conv2,keep_prob=keep_prob)
    
    w_c3=tf.Variable(w_alpha*tf.random_normal([5,5,64,128]))
    b_c3=tf.Variable(w_alpha*tf.random_normal([128]))
    conv3=tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3)
    conv3=batch_norm_layer(conv3,train_phase,scope_bn='bn_3')
    conv3=tf.nn.relu(conv3)
    conv3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3=tf.nn.dropout(conv3,keep_prob=keep_prob)
    
    w_c4=tf.Variable(w_alpha*tf.random_normal([5,5,128,256]))
    b_c4=tf.Variable(w_alpha*tf.random_normal([256]))
    conv4=tf.nn.bias_add(tf.nn.conv2d(conv3,w_c4,strides=[1,1,1,1],padding='SAME'),b_c4)
    conv4=batch_norm_layer(conv4,train_phase,scope_bn='bn_4')
    conv4=tf.nn.relu(conv4)
    conv4=tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv4=tf.nn.dropout(conv4,keep_prob=keep_prob)
    
    #2层全连接层
    w_fully=tf.Variable(w_alpha*tf.random_normal([3*3*256,1024]))
    b_fully=tf.Variable(w_alpha*tf.random_normal([1024]))
    dense=tf.reshape(conv4,[-1,w_fully.get_shape().as_list()[0]])
    dense=tf.nn.relu(tf.matmul(dense,w_fully)+b_fully)
    dense=tf.nn.dropout(dense,keep_prob=keep_prob)
    
    w_out=tf.Variable(w_alpha*tf.random_normal([1024,20]))
    b_out=tf.Variable(b_alpha*tf.random_normal([20]))
    output=tf.matmul(dense,w_out)+b_out
    output=tf.nn.softmax(output)
    return output

def train_handmade_CNN():
    output=handmade_CNN()
    #损失
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=output))
    #优化器
    optimizer=tf.train.AdamOptimizer(learning_rate=lr_base).minimize(loss)
    correction=tf.equal(tf.argmax(Y,1),tf.argmax(output))
    accuracy=tf.reduce_mean(tf.cast(correction,tf.float32))
    
    saver=tf.train.Saver()
    
    init=tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(200):
        batch_x_tr, batch_y_tr = get_batch(x_train,y_train,batch_size=200)
        train_accuracy = accuracy.eval(feed_dict={X:batch_x_tr,Y:batch_y_tr,lr_base:lr_rate,
                        keep_prob:0.5,train_phase:True})
        if i%20==0:
            batch_x_va, batch_y_va = get_batch(x_valid,y_valid,batch_size=100)
            acc=accuracy.run(feed_dict={X:x_valid,Y:y_valid,lr_base:lr_rate,
                                             keep_prob:0.5,train_phase:False})
            print("第{0}步,训练精准率为：{1}".format(i,acc))

train_handmade_CNN()    