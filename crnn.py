#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:56:02 2018

@author: hxj
"""

import tensorflow as tf
from tensorflow.python.training import moving_averages
# Importer and Exporting
# ========

tf.app.flags.DEFINE_string  ('data_path',  './IEMOCAP1.pkl',   'total dataset includes training set, valid set and test set')
tf.app.flags.DEFINE_string  ('checkpoint', './checkpoint/',   'the checkpoint dir')
tf.app.flags.DEFINE_string  ('model_name', 'model.ckpt',      'model name')
tf.app.flags.DEFINE_string  ('pred_name',  './pred0.pkl',        'the test output dir')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  60,         'checkpoint saving interval in seconds')

# Global Constants
# ================

tf.app.flags.DEFINE_float   ('dropout_conv',     1,        'dropout rate for covvolutional layers')
tf.app.flags.DEFINE_float   ('dropout_linear',   1,        'dropout rate for linear layer')
tf.app.flags.DEFINE_float   ('dropout_lstm',     1,        'dropout rate for lstm')
tf.app.flags.DEFINE_float   ('dropout_fully1',   1,        'dropout rate for fully connected layer1')
tf.app.flags.DEFINE_float   ('dropout_fully2',   1,        'dropout rate for fully connected layer1')

#decayed_learning rate
tf.app.flags.DEFINE_float('decay_rate', 0.99, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

#Moving Average
tf.app.flags.DEFINE_integer('decay_steps', 570, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.99, 'the momentum')
tf.app.flags.DEFINE_integer('num_epochs', 30000, 'maximum epochs')
tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters

tf.app.flags.DEFINE_float   ('adam_beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('adam_beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.0001,       'learning rate of Adam optimizer')

# Batch sizes

tf.app.flags.DEFINE_integer ('train_batch_size', 40,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('valid_batch_size',   40,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  40,           'number of elements in a test batch')

tf.app.flags.DEFINE_integer('save_steps', 10, 'the step to save checkpoint')

tf.app.flags.DEFINE_integer('image_height', 300, 'image height')
tf.app.flags.DEFINE_integer('image_width', 40, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')
tf.app.flags.DEFINE_integer('linear_num', 786, 'hidden number of linear layer')
tf.app.flags.DEFINE_integer('seq_len', 150, 'sequence length of lstm')
tf.app.flags.DEFINE_integer('cell_num', 128, 'cell units of the lstm')
tf.app.flags.DEFINE_integer('hidden1', 64, 'number of hidden units of fully connected layer')
tf.app.flags.DEFINE_integer('hidden2', 4, 'number of softmax layer')
tf.app.flags.DEFINE_integer('attention_size', 1, 'attention_size')
tf.app.flags.DEFINE_boolean('attention', False, 'whether to use attention, False mean use max-pooling')

FLAGS = tf.app.flags.FLAGS


class CRNN(object):
    def __init__(self, mode):
        self.mode = mode
        # log Mel-spectrogram
        self.attention = FLAGS.attention
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # emotion label
        self.labels = tf.placeholder(tf.int32, shape=[None, 4])
        # lstm time step
        #self.seq_len = tf.placeholder(tf.int32, [None])
        # l2
        self._extra_train_ops = []
    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='DW',
                                     shape=[filter_size[0], filter_size[1], in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable(name='bais',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides[0], strides[1], 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b) 
    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize[0], ksize[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding='VALID',
                              name='max_pool')
    def _linear(self,x,names,shapes):
        with tf.variable_scope(names):
            weights = tf.get_variable(name='weights',
                                      shape=shapes,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable(name='bias',
                                   shape=shapes[1],
                                   initializer=tf.constant_initializer(0.0))
        return tf.matmul(x,weights) + bias
    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

#                tf.summary.histogram(mean.op.name, mean)
#                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn
    def _batch_norm_wrapper(self, name, inputs, decay = 0.999):
        #batch normalization for fully connected layer
        with tf.variable_scope(name):
            scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
            pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

            if self.mode == 'train':
                batch_mean, batch_var = tf.nn.moments(inputs,[0])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                      pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean, batch_var, beta, scale, FLAGS.epsilon)
            else:
                return tf.nn.batch_normalization(inputs,
                                                 pop_mean, pop_var, beta, scale, FLAGS.epsilon)
    def _attention(self,inputs, attention_size, time_major=False, return_alphas=False):
        
        if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
        # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        #v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
        alphas = tf.nn.softmax(vu)              # (B,T) shape also
        
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas
    def _build_model(self):
        filters = [128, 512]
        filter_size = [5, 3]
        filter_strides = [1, 1]
        pool1_size = [2, 4]
        pool2_size = [1, 2]
        p = 5
        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit-1'):
                x = self._conv2d(self.inputs, 'cnn-1', filter_size, FLAGS.image_channel, filters[0], filter_strides)
                x = self._batch_norm('bn1', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, pool1_size, pool1_size)
#                print x.get_shape()
            with tf.variable_scope('unit-2'):
                x = self._conv2d(x, 'cnn-2',  filter_size, filters[0], filters[1], filter_strides)
                x = self._batch_norm('bn2', x)
                x = self._leaky_relu(x, 0.01)
                x = self._max_pool(x, pool2_size, pool2_size)
#                print x.get_shape()
        with tf.variable_scope('linear'):
            # linear layer for dim reduction
            x = tf.reshape(x,[-1,p*filters[1]])
            x = self._linear(x,'linear1',[p*filters[1],FLAGS.linear_num])
#            print x.get_shape()
        with tf.variable_scope('lstm'):
            x = tf.reshape(x,[-1,FLAGS.seq_len,FLAGS.linear_num])
            
            cell_fw = tf.contrib.rnn.BasicLSTMCell(FLAGS.cell_num, forget_bias=1.0)
            if self.mode == 'train':
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=FLAGS.dropout_lstm)

            cell_bw = tf.contrib.rnn.BasicLSTMCell(FLAGS.cell_num, forget_bias=1.0)
            if self.mode == 'train':
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=FLAGS.dropout_lstm)
            
            # Now we feed `linear` into the LSTM BRNN cell and obtain the LSTM BRNN output.
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                       cell_bw=cell_bw,
                                                                       inputs= x,
                                                                       dtype=tf.float32,
                                                                       time_major=False,
                                                                       scope='LSTM1')
        with tf.variable_scope('time_pooling'):
            if self.attention is not None:
                outputs, alphas = self._attention(outputs, FLAGS.attention_size, return_alphas=True)
            else:
                outputs = tf.concat(outputs,2)
                outputs = tf.reshape(outputs, [-1, FLAGS.seq_len,2*FLAGS.cell_num, 1])
                outputs = self._max_pool(outputs,[FLAGS.seq_len,1],[FLAGS.seq_len,1])
                outputs = tf.reshape(outputs, [-1,2*FLAGS.cell_num])
#            print outputs.get_shape()
        
        with tf.variable_scope('dense'):
            y = self._linear(outputs,'dense-matmul',[2*FLAGS.cell_num,FLAGS.hidden1])
            y = self._batch_norm_wrapper('dense-bn', y)
            y = self._leaky_relu(y, 0.01)
        
        self.logits = self._linear(y,'softmax',[FLAGS.hidden1,FLAGS.hidden2])
            
        
        
    