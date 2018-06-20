# -*- coding: utf-8 -*-
"""
@author: murali.sai
----
Notes:
Fully_Convoluted_RNN for Time-Series Classification using tensorflow library

References:
[1] https://arxiv.org/abs/1709.05206
"""
import os, numpy as np
import tensorflow as tf
from libraries import losses

class Configure_CNN(object):
    def __init__(self):
        # Architecture (CNN layers + RNN Layers + Dense layers)
        self.rnn_units, self.state_activation, self.keep_prob_rnn = [128, 128], tf.nn.tanh, 0.8;
        self.dense_layer_units, self.dense_activation, self.last_activation, self.dropout_rates = [128, 64], tf.nn.relu, tf.nn.relu, [0.1, 0.1];
        assert(len(self.dense_layer_units)==len(self.dropout_rates))
        self.custom_loss = 'categorical_crossentropy' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        if self.custom_loss=='categorical_crossentropy':
            self.last_activation = tf.nn.relu; # Or self.last_activation = None;
        elif self.custom_loss=='cosine_distance' or self.custom_loss=='regression_error':
            self.last_activation = tf.nn.sigmoid;
        # Training and optimization
        self.batch_size, self.n_timesteps, self.n_features, self.n_classes = 128, 457, 1, 10;
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs, self.patience = 200, 50; # Patience: epochs (with no loss improvement) until before terminating the training process
    def create_folders_(self):
        # Directories, Sub-directories, Paths
        self.main_dir = './logs/ts_classification';
        self.model_dir = os.path.join(self.main_dir, 'fc_rnn_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best');
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best');
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
        self.configure_save_path = os.path.join(self.model_dir,'model_configs');
        dirs = [self.main_dir, self.model_dir, self.model_save_training, self.model_save_inference, self.tf_logs, self.images]
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)
class CNN_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_timesteps,self.configure.n_features]) # [batch_size,n_timesteps,n_features]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) # [batch_size,n_classes]
        # Batch Normalization can be performed to alleviate the pain of slow learning due to bad normalization
        with tf.variable_scope('cnn_layers_1'):
	     # [batch_size, 457, 1] --> [batch_size, 457, n_filters_1]
            out_1_1 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='out_1_1')
            out_1_2 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='out_1_2')
            out_1_3 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu, name='out_1_3')
            out_1   = tf.concat([out_1_1,out_1_2,out_1_3], axis=-1, name='out_1') # [batch_size, 457, 48]
            pool_1 = tf.layers.max_pooling1d(inputs=out_1, pool_size=2, strides=2, padding='same', name='max_pool_1') # [batch_size, 229, 48]
            #pool_1 = tf.layers.average_pooling1d(inputs=out_1, pool_size=2, strides=2, padding='same', name='avg_pool_1') # [batch_size, 229, 48]
        with tf.variable_scope('cnn_layers_2'):
	     # [batch_size, 229, n_filters_1] --> [batch_size, 229, n_filters_2]
            out_2_1 = tf.layers.conv1d(inputs=pool_1, filters=8, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='out_2_1')
            out_2_2 = tf.layers.conv1d(inputs=pool_1, filters=6, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='out_2_2')
            out_2_3 = tf.layers.conv1d(inputs=pool_1, filters=4, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu, name='out_2_3')
            out_2 = tf.concat([out_2_1,out_2_2,out_2_3], axis=-1, name='out_2') # [batch_size, 229, 18]
            pool_2 = tf.layers.max_pooling1d(inputs=out_2, pool_size=2, strides=2, padding='same', name='max_pool_2') # [batch_size, 115, 18]
            #pool_2 = tf.layers.average_pooling1d(inputs=out_2, pool_size=2, strides=2, padding='same', name='avg_pool_2') # [batch_size, 115, 18]
        with tf.variable_scope('flatten_layer'):
            flat_out = tf.layers.Flatten(name='flat_out')(pool_2) # Along each batch # [batch_size, 115*18]
        output_ = flat_out;


# <UNDER CONSTRUCTION>