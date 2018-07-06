# -*- coding: utf-8 -*-
"""
@author: murali.sai
----
Notes:
CNN for Time-Series Classification using tensorflow library

References:
[1] https://arxiv.org/pdf/1611.06455.pdf

Future WOrk:
[1] Add batch Normalization & follow the setting: convolutions + batch_normalization + non_linear_activation
[2] Experiments with avg_pooling instead of max_pooling
"""

import os, numpy as np
import tensorflow as tf

from libraries import losses
from models.ts_classification.global_params import global_params

class Configure_CNN(object):
    def __init__(self):
        # Architecture Parameters (CNN Layers + Dense Layers)
        self.dense_layer_units = [256, 128, 64];
        self.dropout_rates = [0.25, 0.125, 0.1];
        self.dense_activation = tf.nn.relu;
        self.last_activation = None;
        # Load global (initialized) parameters
        gp = global_params();
        # Training and optimization
        self.batch_size = gp.batch_size;
        self.n_timesteps = gp.n_timesteps;
        self.n_features = gp.n_features;
        self.n_classes = gp.n_classes;
        self.max_gradient_norm = gp.max_gradient_norm;
        self.learning_rate = gp.learning_rate;
        self.lr_decay_steps = gp.lr_decay_steps;
        self.lr_decay_mag = gp.lr_decay_mag;
        self.n_epochs = gp.n_epochs;
        self.patience = gp.patience;
        self.parent_folder = gp.parent_folder;
        self.custom_loss = gp.loss_function;
        # last_activation
        if self.custom_loss=='categorical_crossentropy':
            self.last_activation = tf.nn.relu;
        elif self.custom_loss=='cosine_distance' or self.custom_loss=='regression_error':
            self.last_activation = tf.nn.sigmoid;
        # Validate
        self.validate_();
    def validate_(self):
        assert(len(self.dense_layer_units)==len(self.dropout_rates))
        assert(self.last_activation!=None)
    def create_folders_(self): # Directories, Sub-directories, Paths
        print('parent Folder set as: {} If needed, please change it relative to you current path'.format(self.parent_folder))
        self.main_dir = os.path.join(self.parent_folder, 'logs', 'ts_classification');
        self.model_dir = os.path.join(self.main_dir, 'cnn_tf_'+self.custom_loss);
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
        '''
        # fil_a_b_c implies a:cnn_layer, b=kernel_size, c=n^th_level; similar termilology for out_a_b_c
        '''
        self.weights_initializer = tf.contrib.layers.xavier_initializer();
        # Conv Layer 1
        conv_1_input = self.x_;
        in_channels = self.configure.n_features;
        [fil_1_3_1, fil_1_3_2, fil_1_5_1, fil_1_7_1] = [16,16,8,4]
        with tf.variable_scope('cnn_layers_1_3'):
            w_1_3_1 = tf.get_variable("w_1_3_1", shape=[3, in_channels, fil_1_3_1], initializer=self.weights_initializer);
            b_1_3_1 = self.init_bias([fil_1_3_1], 'b_1_3_1')
            conv_1_3_1 = tf.nn.conv1d(conv_1_input, w_1_3_1, stride=1, padding='SAME', data_format='NWC') + b_1_3_1;
            w_1_3_2 = tf.get_variable("w_1_3_2", shape=[3, fil_1_3_1, fil_1_3_2], initializer=self.weights_initializer);
            b_1_3_2 = self.init_bias([fil_1_3_2], 'b_1_3_2')
            conv_1_3_2 = tf.nn.conv1d(conv_1_3_1, w_1_3_2, stride=1, padding='SAME', data_format='NWC') + b_1_3_2;
        with tf.variable_scope('cnn_layers_1_5'):
            w_1_5_1 = tf.get_variable("w_1_5_1", shape=[5, in_channels, fil_1_5_1], initializer=self.weights_initializer);
            b_1_5_1 = self.init_bias([fil_1_5_1], 'b_1_5_1')
            conv_1_5_1 = tf.nn.conv1d(conv_1_input, w_1_5_1, stride=1, padding='SAME', data_format='NWC') + b_1_5_1;
        with tf.variable_scope('cnn_layers_1_7'):
            w_1_7_1 = tf.get_variable("w_1_7_1", shape=[3, in_channels, fil_1_7_1], initializer=self.weights_initializer);
            b_1_7_1 = self.init_bias([fil_1_7_1], 'b_1_7_1')
            conv_1_7_1 = tf.nn.conv1d(conv_1_input, w_1_7_1, stride=1, padding='SAME', data_format='NWC') + b_1_7_1;
        conv_1_output   = tf.concat([conv_1_3_2,conv_1_5_1,conv_1_7_1], axis=-1, name='conv_1_output') # [batch_size, n_timesteps, n_filters_1]
        conv_1_output_pool = tf.layers.max_pooling1d(inputs=conv_1_output, pool_size=2, strides=2, padding='same', name='conv_1_output_pool') # [batch_size, n_timesteps/2, n_filters_1]
        # Conv Layer 2
        conv_2_input = conv_1_output_pool;
        in_channels = fil_1_3_2+fil_1_5_1+fil_1_7_1;
        [fil_2_3_1, fil_2_3_2, fil_2_5_1, fil_2_5_2] = [64,64,64,64]
        with tf.variable_scope('cnn_layers_2_3'):
            w_2_3_1 = tf.get_variable("w_2_3_1", shape=[3, in_channels, fil_2_3_1], initializer=self.weights_initializer);
            b_2_3_1 = self.init_bias([fil_2_3_1], 'b_2_3_1')
            conv_2_3_1 = tf.nn.conv1d(conv_2_input, w_2_3_1, stride=1, padding='SAME', data_format='NWC') + b_2_3_1;
            w_2_3_2 = tf.get_variable("w_2_3_2", shape=[3, fil_2_3_1, fil_2_3_2], initializer=self.weights_initializer);
            b_2_3_2 = self.init_bias([fil_2_3_2], 'b_2_3_2')
            conv_2_3_2 = tf.nn.conv1d(conv_2_3_1, w_2_3_2, stride=1, padding='SAME', data_format='NWC') + b_2_3_2;
        with tf.variable_scope('cnn_layers_2_5'):
            w_2_5_1 = tf.get_variable("w_2_5_1", shape=[5, in_channels, fil_2_5_1], initializer=self.weights_initializer);
            b_2_5_1 = self.init_bias([fil_2_5_1], 'b_2_5_1')
            conv_2_5_1 = tf.nn.conv1d(conv_2_input, w_2_5_1, stride=1, padding='SAME', data_format='NWC') + b_2_5_1;
            w_2_5_2 = tf.get_variable("w_2_5_2", shape=[5, fil_2_5_1, fil_2_5_2], initializer=self.weights_initializer);
            b_2_5_2 = self.init_bias([fil_2_5_2], 'b_2_5_2')
            conv_2_5_2 = tf.nn.conv1d(conv_2_5_1, w_2_5_2, stride=1, padding='SAME', data_format='NWC') + b_2_5_2;
        conv_2_output   = tf.concat([conv_2_3_2,conv_2_5_2], axis=-1, name='conv_2_output')
        conv_2_output_pool = tf.layers.max_pooling1d(inputs=conv_2_output, pool_size=2, strides=2, padding='same', name='conv_2_output_pool') # [batch_size, n_timesteps/4, n_filters_2]
        # Conv Layer 3; Kind of dimensional reduction using 1x1 conv
        conv_3_input = conv_2_output_pool;
        in_channels = fil_2_3_2+fil_2_5_2;
        [fil_3_1_1] = [48]
        with tf.variable_scope('cnn_layers_3_1'):
            w_3_1_1 = tf.get_variable("w_3_1_1", shape=[1, in_channels, fil_3_1_1], initializer=self.weights_initializer);
            b_3_1_1 = self.init_bias([fil_3_1_1], 'b_3_1_1')
            conv_3_1_1 = tf.nn.conv1d(conv_3_input, w_3_1_1, stride=1, padding='SAME', data_format='NWC') + b_3_1_1;
        conv_3_output = conv_3_1_1;
        conv_3_output_pool = tf.layers.max_pooling1d(inputs=conv_3_output, pool_size=2, strides=2, padding='same', name='conv_3_output_pool') # [batch_size, n_timesteps/8, n_filters_3]
        with tf.variable_scope('flatten_layer'):
            flat_out = tf.layers.Flatten(name='flat_out')(conv_3_output_pool) # [batch_size, n_timesteps/8 * n_filters_3]
        output_ = flat_out;
        with tf.variable_scope('multi_dense_layers'):
            for i, units in enumerate(self.configure.dense_layer_units):
                output_ = tf.layers.dense(inputs=output_, units=units, activation=self.configure.dense_activation, name='dense_{}'.format(i))
                output_ = tf.layers.dropout(output_, rate=self.configure.dropout_rates[i], training=self.training, name='dropout_{}'.format(i))
            self.preds = tf.layers.dense(inputs=output_, units=self.configure.n_classes, activation=self.configure.last_activation, name='predictions')
        with tf.variable_scope('loss_and_optimizer'):
            # 1. Loss function
            self.loss = (tf.reduce_sum(getattr(losses, self.configure.custom_loss)(self.y_,self.preds))/tf.cast(tf.shape(self.x_)[0],tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_,1), tf.argmax(self.preds,1)), tf.float32), name='accuracy')
            # 2. Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.configure.max_gradient_norm)
            # 3. Set learning Rate: Exponential Decay or a constant value
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = tf.train.exponential_decay(self.configure.learning_rate, self.global_step, self.configure.lr_decay_steps, self.configure.lr_decay_mag)
            #self.lr = self.configure.learning_rate;
            # 4. Update weights and biases i.e trainable parameters
            # We need to pass global_step as a parameter to the optimizer so it knows to increment global_step by one with each training step
            self.update_step = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            #self.update_step = tf.train.RMSPropOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    def init_bias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape = shape), name=name)
    def make_data_for_batch_training_CNN(self, x_data, y_data):
        # Inputs:: x_data:[n_samples,n_timesteps,n_features] y_data:[n_samples,n_classes]
        # Outputs:: x_data_new:[n_batches,batch_size,n_timesteps,n_features] y_data_new:[n_batches,batch_size,n_classes]
        # Zero Mean Substraction can be performed additionally
        assert(x_data.shape[0]==y_data.shape[0]);
        n_rows = x_data.shape[0];
        x_data_new, y_data_new = [], [];
        for i in np.arange(0,n_rows-n_rows%self.configure.batch_size,self.configure.batch_size):
            x_data_new.append(x_data[i:i+self.configure.batch_size])
            y_data_new.append(y_data[i:i+self.configure.batch_size])
        if n_rows%self.configure.batch_size!=0: # Implies left over samples must be added into a last batch
            x_data_new.append(x_data[-self.configure.batch_size:])
            y_data_new.append(y_data[-self.configure.batch_size:])
        x_data_new = np.stack(x_data_new);
        y_data_new = np.stack(y_data_new);
        return x_data_new, y_data_new






#        with tf.variable_scope('cnn_layers_1'):
#	     # [batch_size, 457, 1] --> [batch_size, 457, n_filters_1]
#            out_1_1 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='out_1_1')
#            out_1_2 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='out_1_2')
#            out_1_3 = tf.layers.conv1d(inputs=self.x_, filters=16, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu, name='out_1_3')
#            out_1   = tf.concat([out_1_1,out_1_2,out_1_3], axis=-1, name='out_1') # [batch_size, 457, 48]
#            pool_1 = tf.layers.max_pooling1d(inputs=out_1, pool_size=2, strides=2, padding='same', name='max_pool_1') # [batch_size, 229, 48]
#        with tf.variable_scope('cnn_layers_2'):
#	     # [batch_size, 229, n_filters_1] --> [batch_size, 229, n_filters_2]
#            out_2_1 = tf.layers.conv1d(inputs=pool_1, filters=8, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='out_2_1')
#            out_2_2 = tf.layers.conv1d(inputs=pool_1, filters=6, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu, name='out_2_2')
#            out_2_3 = tf.layers.conv1d(inputs=pool_1, filters=4, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu, name='out_2_3')
#            out_2 = tf.concat([out_2_1,out_2_2,out_2_3], axis=-1, name='out_2') # [batch_size, 229, 18]
#            pool_2 = tf.layers.max_pooling1d(inputs=out_2, pool_size=2, strides=2, padding='same', name='max_pool_2') # [batch_size, 115, 18]
#










