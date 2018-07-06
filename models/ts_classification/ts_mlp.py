# -*- coding: utf-8 -*-
"""
@author: murali.sai
----
Notes:
MLP for Time-Series Classification using tensorflow library
"""
import os, numpy as np
import tensorflow as tf

from libraries import losses
from models.ts_classification.global_params import global_params

class Configure_MLP(object):
    def __init__(self):
        # Architecture Parameters (Dense Layers)
        self.dense_layer_units = [512, 256, 256, 128, 64, 32];
        self.dropout_rates = [0, 0, 0.1, 0.1, 0.1, 0.1];
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
        self.model_dir = os.path.join(self.main_dir, 'mlp_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best')
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best')
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
        self.configure_save_path = os.path.join(self.model_dir,'model_configs');
        dirs = [self.main_dir, self.model_dir, self.model_save_training, self.model_save_inference, self.tf_logs, self.images]
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.mkdir(dir_)

class MLP_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_timesteps]) # [batch_size,n_timesteps]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) # [batch_size,n_classes]
        output_ = self.x_;
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
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step') # global_step just keeps track of the number of batches seen so far
            #self.lr = tf.train.exponential_decay(self.configure.learning_rate, self.global_step, self.configure.max_global_steps_assumed, 0.1)
            self.lr = self.configure.learning_rate;
            # 4. Update weights and biases i.e trainable parameters
            self.update_step = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            #self.update_step = tf.train.RMSPropOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    def make_data_for_batch_training_MLP(self, x_data, y_data):
        # Inputs:: x_data:[n_samples,n_timesteps] y_data:[n_samples,n_classes]
        # Outputs:: x_data_new:[n_batches,batch_size,n_timesteps] y_data_new:[n_batches,batch_size,n_classes]
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
