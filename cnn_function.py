#Functions for CNN
import os
#Setting environment variable for GPU (Set to -1 for CPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

def init_weight(shape):
    '''
    Initialize weights for CNN and DNN layers (Return a TF variable)
    '''
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    '''
    Initialize bias for CNN and DNN layers (Returns a TF variable)
    '''
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x,W):
    '''
    Convolute a 2D matrix
    x --> Input tensor [batch,H,W,Channels/Color layers] (4 dimensions)
    W --> Kernel with 4 dimensions [filter H, filter W, Channel IN, Channel OUT]
    
    '''
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

def max_pooling_2by2(x):
    '''
    2x2 pooling layer
    x--> Input tensor [batch,H,W,c]
    '''
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def conv_layer(x,shape,name=None):
    '''
    Returns convolution layer
    '''
    W = init_weight(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(x,W)+b,name=name)

def dnn_layer(input_layer,size,name=None):
    '''
    Returns DNN layer
    '''
    input_size = int(input_layer.get_shape()[1])
    W = init_weight([input_size,size])
    b = init_bias([size])
    return tf.add(tf.matmul(input_layer,W),b,name=name)