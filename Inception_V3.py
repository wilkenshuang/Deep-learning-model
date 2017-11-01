# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:40:13 2017

@author: wilkenshuang
"""

import tensorflow as tf

slim=tf.contrib.slim
trunc_normal=tf.truncated_normal_initializer(0.0,stddev)

def inception_v3_parameters(weight_decay=0.0004,stddev=0.1,batch_norm_var_collection='moving_vars'):
    batch_norm_params={'decay':0.997,'epsilon':0.001,'updates_collections':tf.GraphKeys.UPDATE_OPS,
                       'variable_collections':{'beta':None,'gamma':None,
                                               'moving_mean':[batch_norm_var_collection],
                                               'moving_variance':[batch_norm_var_collection]}}
    with slim.arg_scope([slim.conv2d],weights_initializer=trunc_normal(stddev),
                        activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as sc:
        return sc

def inception_v3_base(inputs,scope=None):
    end_points={}
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
    # 输入图片大小为299*299*3    
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
        # 对三个参数设置默认值
            net=slim.conv2d(inputs,32,[3,3],stride=2,scope='conv2d_2a_3x3')# 149*149*32
            net=slim.conv2d(net,32,[3,3],padding='SAME',scope='conv2d_2b_3x3')# 147*147*32
            net=slim.max_pool2d(net,[3,3],stride=2,scope='maxpool_3a_3x3')# 73*73*32
            net=slim.conv2d(net,64,[1,1],scope='conv2d_3b_1x1')# 73*73*64
            net=slim.conv2d(net,128,[3,3],scope='con2d_4a_3x3')# 71*71*128
            net=slim.max_pool2d(net,[3,3],stride=2,scope='maxpool_5a_3x3')# 35*35*128
            # 上面部分代码一共有5个卷积层，2个池化层，实现了对图片数据的尺寸压缩，并对图片特征进行了抽象

            
        # Inception blocks
        # 三个连续的inception模块组，每个模块组内有多个小的模块，包括卷积、最大池化
        # 该部分就是inception最大的特色，每个模块组非常相似，但其内部的小模块的数量有些许不一样
        # 设置所有模块组的默认参数,将所有卷积层、最大池化、平均池化层步长都设置为1
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
            
            # 第一个模块组包含了三个结构类似的Inception Module
            with tf.variable_scope('mixed_1a'): # 第一个Inception Module名称。Inception Module有四个分支
                with tf.variable_scope('brand_0'):# 第一个分支64通道的1*1卷积
                    branch_0=slim.conv2d(net,64,[1,1],scope='con2d_0a_1x1')
                with tf.variable_scope('brand_1'): # 第二个分支48通道1*1卷积，链接一个64通道的5*5卷积
                    branch_1=slim.conv2d(net,48,[1,1],scope='conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='conv2d_0b_5x5')
                with tf.variable_scope('branch_2'):# 第三个分支64通道1*1卷积，链接一个96通道的3*3卷积
                    branch_2=slim.conv2d(net,64,[1,1],scope='conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='conv2d_0c_3x3')
                with tf.variable_scope('branch_3'):# 第四个分支为3*3的平均池化，连接32通道的1*1卷积
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,32,[1,1],scope='conv2d_0b_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
                # 这里因为采用的padding（填补）方式是same，所以图片的大小不会改变，仍然是35*35，但因为
                # 所有通道的连接，输出通道数变成了64+64+96+32=256个，所以最后输出的tensor大小为35*35*256.
                
            with tf.variable_scope('mixed_1b'):
                with tf.variable_scope('brach_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('brand_1'): 
                    branch_1=slim.conv2d(net,48,[1,1],scope='conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='conv2d_0b_5x5')
                with tf.variable_scope('brand_2'): 
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='avgpool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='conv2d_0b_1x1')
                net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
                # output_tensor:35*35*288
                
            with tf.variable_scope('mixed_1c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # output_tensor:35*35*288
            
            # 第二个Inception模块组
            with tf.variable_scope('mixed_2a'):
                with tf.variable_scope('branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,padding='VALID', 
                                           scope='Conv2d_1a_1x1')# 图片会被压缩
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,padding='VALID',
                                           scope='Conv2d_1a_1x1') 
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3) 
                # output_tensor:17*17*768
                
            with tf.variable_scope('Mixed_2b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7') 
                    # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'): 
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1') 
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # output_tensor:17*17*768
            
            # 网络每经过一个inception module，即使输出尺寸不变，但是特征都相当于被重新精炼了一遍，
            # 其中丰富的卷积和非线性化对提升网络性能帮助很大。
            with tf.variable_scope('Mixed_2c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7') 
                    # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'): 
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1') 
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # output_tensor:17*17*768
            
            with tf.variable_scope('Mixed_2d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7') 
                    # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'): 
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1') 
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # output_tensor:17*17*768
                
            with tf.variable_scope('Mixed_2e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7') 
                    # 串联1*7卷积和7*1卷积合成7*7卷积，减少了参数，减轻了过拟合
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'): 
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1') # 反复将7*7卷积拆分
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1') 
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # output_tensor:17*17*768
                
            return net,end_points
            #Inception V3网络的核心部分，即卷积层部分就完成了
'''
设计inception net的重要原则是图片尺寸不断缩小，inception模块组的目的都是将空间结构简化，同时将空间信息转化为
高阶抽象的特征信息，即将空间维度转为通道的维度。降低了计算量。Inception Module是通过组合比较简单的特征
抽象（分支1）、比较比较复杂的特征抽象（分支2和分支3）和一个简化结构的池化层（分支4），一共四种不同程度的
特征抽象和变换来有选择地保留不同层次的高阶特征，这样最大程度地丰富网络的表达能力。
'''

# 全局平均池化、Softmax和Auxiliary Logits
def inception_v3(inputs,
                 num_classes=1000, # 最后需要分类的数量（比赛数据集的种类数）
                 is_training=True, # 标志是否为训练过程，只有在训练时Batch normalization和Dropout才会启用
                 dropout_keep_prob=0.8, # 节点保留比率
                 prediction_fn=slim.softmax, # 最后用来分类的函数
                 spatial_squeeze=True, # 参数标志是否对输出进行squeeze操作（去除维度数为1的维度，比如5*3*1转为5*3）
                 reuse=None, # 是否对网络和Variable进行重复使用
                 scope='InceptionV3'): # 包含函数默认参数的环境

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], # 定义参数默认值
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], # 定义标志默认值
                        is_training=is_training):
      # 拿到最后一层的输出net和重要节点的字典表end_points
      net, end_points = inception_v3_base(inputs, scope=scope) # 用定义好的函数构筑整个网络的卷积部分

      # Auxiliary Head logits作为辅助分类的节点，对分类结果预测有很大帮助
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'): # 将卷积、最大池化、平均池化步长设置为1
        aux_logits = end_points['Mixed_6e'] # 通过end_points取到Mixed_6e
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
              aux_logits, [5, 5], stride=3, padding='VALID', # 在Mixed_6e之后接平均池化。压缩图像尺寸
              scope='AvgPool_1a_5x5')
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1], # 卷积。压缩图像尺寸。
                                   scope='Conv2d_1b_1x1')

          # Shape of feature map before the final layer.
          aux_logits = slim.conv2d(
              aux_logits, 768, [5,5],
              weights_initializer=trunc_normal(0.01), # 权重初始化方式重设为标准差为0.01的正态分布
              padding='VALID', scope='Conv2d_2a_5x5')
          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001), # 输出变为1*1*1000
              scope='Conv2d_2b_1x1')
          if spatial_squeeze: # tf.squeeze消除tensor中前两个为1的维度。
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits # 最后将辅助分类节点的输出aux_logits储存到字典表end_points中

      # 处理正常的分类预测逻辑
      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                              scope='AvgPool_1a_8x8')
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, # 输出通道数1000
                             normalizer_fn=None, scope='Conv2d_1c_1x1') # 激活函数和规范化函数设为空
        if spatial_squeeze: # tf.squeeze去除输出tensor中维度为1的节点
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions') # Softmax对结果进行分类预测
  return logits, end_points # 最后返回logits和包含辅助节点的end_points

            
                     