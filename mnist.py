# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 21:45:27 2017

@author: wilkenshuang
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess=tf.InteractiveSession()

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[10])
x_image=tf.reshape(x,[-1,28,28,1])

def weight_variable(inputs):
    initial=tf.truncated_normal(inputs,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(inputs):
    initial=tf.truncated_normal(inputs,stddev=0.1)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME'))

def max_pool_2x2(x):
    return tf.nn.relu(tf.nn.max_pool(x,[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
tmp_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
res_pool1=tf.nn.relu(max_pool_2x2(tmp_conv1))

W_conv2=weight_variable([5,5,32,64])
b_conv1=bias_variable([64])
tmp_conv2=tf.nn.relu(conv2d(res_pool1,W_conv2)+b_conv1)
res_pool1=max_pool_2x2(tmp_conv2)

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
x_flat=tf.reshape(res_pool1,[-1,7*7*64])
res_fc1=tf.nn.relu(tf.matmul(x_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
res_fc1_drop=tf.nn.dropout(res_fc1,keep_prob=keep_prob)

W_output=weight_variable([1024,10])
b_output=bias_variable([10])
output=tf.matmul(res_fc1_drop,W_output)+b_output

entropy_loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output)
optimizer=tf.train.AdamOptimizer(learning_rate=0.005).minimize(entropy_loss)

correct=tf.equal(tf.argmax(y,1),tf.argmax(output,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch=mnist.train.next_batch(50)
    if i%10==0:
        accuracy=accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:0.8})
        print("循环：%d, 训练精度: %g" %(i,accuracy))
    sess.run(feed_dict={x:batch[0],y:batch[1],keep_prob:0.5})
        

