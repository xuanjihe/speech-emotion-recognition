#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:34:21 2017
@author: hxj
"""



from attention import attention
import cPickle
import tensorflow as tf
import math
import numpy as np
import os
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
epsilon = 1e-3

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
def load_data():
    f = open('./CASIA_40_delta.pkl','rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = cPickle.load(f)
    #train_data,train_label,test_data,test_label,valid_data,valid_label = cPickle.load(f)
    return train_data,train_label,test_data,test_label,valid_data,valid_label
def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def build_model(inputX, is_training,keep_prob):
    # 3 2-D convolution layers
    L1 = 256
    L2 = 512
    L3 = 512
    Li1 = 768
    F1 = 64
    F2 = 6
    p = 5
    cell_units1 = 128
    timesteps = 200
    ATTENTION_SIZE = 1
    layer1_filter = tf.get_variable('layer1_filter', shape=[5, 3, 3, L1], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer1_bias = tf.get_variable('layer1_bias', shape=[L1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer1_stride = [1, 1, 1, 1]
    layer2_filter = tf.get_variable('layer2_filter', shape=[5, 3, L1, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer2_bias = tf.get_variable('layer2_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer2_stride = [1, 1, 1, 1]
    layer3_filter = tf.get_variable('layer3_filter', shape=[5, 3, L2, L3], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer3_bias = tf.get_variable('layer3_bias', shape=[L3], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer3_stride = [1, 1, 1, 1]
    
    linear1_weight = tf.get_variable('linear1_weight', shape=[p*L2,Li1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    linear1_bias = tf.get_variable('linear1_bias', shape=[Li1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
 
    fully1_weight = tf.get_variable('fully1_weight', shape=[2*cell_units1,F1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully1_bias = tf.get_variable('fully1_bias', shape=[F1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    fully2_weight = tf.get_variable('fully2_weight', shape=[F1,F2], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully2_bias = tf.get_variable('fully2_bias', shape=[F2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer1 = tf.nn.conv2d(inputX, layer1_filter, layer1_stride, padding='SAME')
    layer1 = tf.nn.bias_add(layer1,layer1_bias)
    #layer1 = tf.layers.batch_normalization(layer1, training=is_training)
    #layer1 = Batch_Normalization(layer1, training=is_training, scope='layer1_batch')
    layer1 = leaky_relu(layer1, 0.01)
    #layer1 = Batch_Normalization(layer1, training=is_training, scope='layer1_batch')
    #print layer1.get_shape()
    layer1 = tf.nn.max_pool(layer1,ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID', name='max_pool')
    #print layer1.get_shape()
    layer1 = tf.contrib.layers.dropout(layer1, keep_prob=keep_prob, is_training=is_training)
    #layer1 = tf.reshape(layer1,[-1,timesteps,L1*p])
    
    layer2 = tf.nn.conv2d(layer1, layer2_filter, layer2_stride, padding='SAME')
    layer2 = tf.nn.bias_add(layer2,layer2_bias)
    #layer1 = tf.layers.batch_normalization(layer1, training=is_training)
    
    layer2 = leaky_relu(layer2, 0.01)
    #print layer2.get_shape()
    #layer2 = Batch_Normalization(layer2, training=is_training, scope='layer1_batch')
    layer2 = tf.nn.max_pool(layer2,ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID', name='max_pool')
    #print layer2.get_shape()
    layer2 = tf.contrib.layers.dropout(layer2, keep_prob=keep_prob, is_training=is_training)
    layer2 = tf.reshape(layer2,[-1,timesteps,L2*p])
    
    
    layer2 = tf.reshape(layer2, [-1,p*L2])
    
    #layer1 = tf.reshape(layer1,[-1,p*L1])
    linear1 = tf.matmul(layer2,linear1_weight) + linear1_bias
    linear1 = batch_norm_wrapper(linear1,is_training)
    linear1 = leaky_relu(linear1, 0.01)
    #linear1 = batch_norm_wrapper(linear1,is_training)
    linear1 = tf.reshape(linear1, [-1, timesteps, Li1])
    
    
    '''
    #adding gru cell
    gru_bw_cell1 = tf.nn.rnn_cell.GRUCell(cell_units)
    #if is_training is not None:
    #    gru_bw_cell1 = tf.contrib.rnn.DropoutWrapper(cell=gru_bw_cell1, output_keep_prob=keep_prob)
    # Forward direction cell: (if else required for TF 1.0 and 1.1 compat)
    gru_fw_cell1 = tf.nn.rnn_cell.GRUCell(cell_units)
    #if is_training is not None:
    #    gru_fw_cell1 = tf.contrib.rnn.DropoutWrapper(cell=gru_fw_cell1, output_keep_prob=keep_prob)
    
    '''
    # Define lstm cells with tensorflow
    # Forward direction cell
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units1, forget_bias=1.0)
    # Backward direction cell
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units1, forget_bias=1.0)
    
    '''
    # Define lstm cells with tensorflow
    # Forward direction cell
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    if is_training is not None:
        gru_fw_cell1 = tf.contrib.rnn.DropoutWrapper(cell=gru_fw_cell1, output_keep_prob=keep_prob)
    # Backward direction cell
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    if is_training is not None:
        gru_bw_cell1 = tf.contrib.rnn.DropoutWrapper(cell=gru_bw_cell1, output_keep_prob=keep_prob)
    '''
    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell1,
                                                             cell_bw=gru_bw_cell1,
                                                             inputs= linear1,
                                                             dtype=tf.float32,
                                                             time_major=False,
                                                             scope='LSTM1')
    '''
    outputs1 = tf.concat(outputs1,2)
     # Forward direction cell
    gru_fw_cell2 = tf.contrib.rnn.BasicLSTMCell(cell_units2, forget_bias=1.0)
    # Backward direction cell
    gru_bw_cell2 = tf.contrib.rnn.BasicLSTMCell(cell_units2, forget_bias=1.0)
    # Now we feed `layer_3` into the LSTM BRNN cell and obtain the LSTM BRNN output.
    outputs, output_states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell2,
                                                             cell_bw=gru_bw_cell2,
                                                             inputs= outputs1,
                                                             dtype=tf.float32,
                                                             time_major=False,
                                                             scope='LSTM2')
    '''
    #time_major=false,tensor的shape为[batch_size, max_time, depth]。实验中使用tf.concat(outputs, 2)将其拼接
    
    outputs = tf.concat(outputs1,2)
    outputs = tf.reshape(outputs, [-1, timesteps,2*cell_units1, 1])
    gru = tf.nn.max_pool(outputs,ksize=[1,timesteps,1,1], strides=[1,timesteps,1,1], padding='VALID', name='max_pool')
    gru = tf.reshape(gru, [-1,2*cell_units1])    
    '''
    # Attention layer
    gru, alphas = attention(outputs1, ATTENTION_SIZE, return_alphas=True)
    ''' 
    
    fully1 = tf.matmul(gru,fully1_weight) + fully1_bias
    #fully1 = batch_norm_wrapper(fully1,is_training)
    fully1 = leaky_relu(fully1, 0.01)
    #fully1 = batch_norm_wrapper(fully1,is_training) 
    fully1 = tf.nn.dropout(fully1, keep_prob)
    
    
    Ylogits = tf.matmul(fully1, fully2_weight) + fully2_bias
    #Ylogits = tf.nn.softmax(Ylogits)
    '''
    fully2 = tf.matmul(fully1,fully2_weight) + fully2_bias  
    fully2 = leaky_relu(fully2, 0.01)
    #fully2 = batch_norm_wrapper(fully2,is_training) 
    Ylogits = tf.matmul(fully2, fully3_weight) + fully3_bias
    #Ylogits = tf.nn.softmax(Ylogits)
    '''
    return Ylogits
    
def train_op(norm):
    STEPS = 50000
    batch_size = 60
    grad_clip = 5
    MODEL_SAVE_PATH = "./model2/"
    MODEL_NAME = "model.ckpt"
    X = tf.placeholder(tf.float32, shape=[None, 300,40,3])
    Y = tf.placeholder(tf.int32, shape=[None, 4])
    is_training = tf.placeholder(tf.bool)
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = build_model(X, is_training, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels =  Y, logits =  Ylogits)
    cost = tf.reduce_mean(cross_entropy)
    #train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    var_trainable_op = tf.trainable_variables()
    if norm == -1:
        # not apply gradient clipping
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)            
    else:
        # apply gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, var_trainable_op), grad_clip)
        opti = tf.train.AdamOptimizer(lr)
        train_op = opti.apply_gradients(zip(grads, var_trainable_op))
    correct_pred = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   
    saver=tf.train.Saver(tf.global_variables())
    
    train_data,train_label,test_data,test_label,valid_data,valid_label = load_data()
    train_label = dense_to_one_hot(train_label,len(np.unique(train_label)))
    test_label = dense_to_one_hot(test_label,len(np.unique(test_label)))
    valid_label = dense_to_one_hot(valid_label,len(np.unique(valid_label)))
    max_learning_rate = 0.0001
    min_learning_rate = 0.000001
    decay_speed = 1600
    dataset_size = train_data.shape[0]
    # init
    init = tf.global_variables_initializer()
    best_acc = 0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(STEPS):
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
            start = (i * batch_size) % dataset_size
            end = min(start+batch_size, dataset_size)
            if i % 5 == 0:
                loss, train_acc = sess.run([cost,accuracy],feed_dict = {X:valid_data, Y:valid_label,is_training:False, keep_prob:1})
                test_acc = sess.run(accuracy, feed_dict = {X:test_data, Y:test_label, is_training:False, keep_prob:1})
                if test_acc > best_acc:
                    best_acc = test_acc
                print "After %5d trainging step(s), validation cross entropy is %2.2g, validation accuracy is %3.2g, test accuracy is %3.2g, the best accuracy is %3.2g" %(i, loss, train_acc, test_acc, best_acc)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step = i)
            sess.run(train_op, feed_dict={X:train_data[start:end,:,:,:], Y:train_label[start:end,:],
                                            is_training:True, keep_prob:1, lr:learning_rate})
                                    
if __name__=='__main__':
    train_op(1)
