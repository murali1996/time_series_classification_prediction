# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:38:56 2018
@author: murali.sai
"""
#%% Libraries
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization, Dropout
from keras.layers import LSTM, GRU, Masking
from keras.optimizers import SGD, RMSprop, Adam

import datasets.data_reader
from helpers import train_test_split, progressBar
from losses import categorical_crossentropy, cosine_distance, regression_error, hinge_loss

# %%TODO:
# 1. Implement custom objectives:
#    cross-entropy (done), cosine distance,
#    regression error, hinge loss
# 2. Implement network architectures:
#    MLP (done), RNN, CNN
# 3. For each architecture, train a classifier using
#    cross-entropy, cosine distance, and hinge loss.
#    Plot loss/accuracy as a function of epoch on one graph.
#    (9 models total)

#%% PART 1
#%% 1.1 Load data
x, y = datasets.data_reader.read_clean_dataset(summary=True)
y = datasets.data_reader.one_hot(y)
x_train, y_train, x_test, y_test = train_test_split(x, y)
feat_dim, out_dim = x.shape[1], y.shape[1]
#%% 1.2 Basic Data Analysis
## Shapes
#x.shape
#for data in [x_train, y_train, x_test, y_test]:
#    print(data.shape)
## Range of values
#print('max and min in x_train: {}, {}'.format(np.max(x_train),np.min(x_train)))
#print('max and min in x_test: {}, {}'.format(np.max(x_test),np.min(x_test)))
#print('any NaNs? {}, {}'.format(np.sum(np.isnan(x_train)),np.sum(np.isnan(x_test))))
## Plots
#for label in range(y_train.shape[-1]):
#    labelled = x_train[np.where(y_train[:,label]==1)[0],:]
#    fig_rows, fig_cols = 2, 2
#    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(16, 5))
#    row_inds = np.random.choice(labelled.shape[0],fig_rows*fig_cols,replace=False);
#    for fig_row in range(fig_rows):
#        for fig_col in range(fig_cols):
#            ax[fig_row,fig_col].plot(labelled[row_inds[fig_row*fig_cols+fig_col],:]);
#            ax[fig_row,fig_col].set_title('{}'.format(row_inds[fig_row*fig_cols+fig_col]));
#    for axx in ax.flat: # Hide x labels and tick labels for top plots and y ticks for right plots.
#        axx.label_outer()
#    fig.suptitle('Label: {}'.format(label))
#%% 1.3 RNN for Time-Series Classification
class Configure(object):
    def __init__(self):
        self.rnn_units, self.state_activation = [128, 128], tf.nn.tanh;
        self.batch_size, self.n_classes = 128, 10;
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs = 10;
class rnn_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.x_ = tf.placeholder(tf.float32, shape=[None,None,1]) #[batch_size,past_time_steps,num_fetaures]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) #[batch_size,n_classes]
        with tf.variable_scope('multi_layer_rnn'):
            rnn_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, activation=self.configure.state_activation) \
                         for n in self.configure.rnn_units]
            rnn_cells_stack = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
            outputs, state = tf.nn.dynamic_rnn(rnn_cells_stack, self.x_, dtype = tf.float32)
            self.output = outputs[:,-1,:] # Of shape [batch_size,self.configure.rnn_units[-1]]
        with tf.variable_scope('dense_layers'):
            self.weights = {'w_1':self.init_weights([self.configure.rnn_units[-1],256],'w_1'),
                            'w_2':self.init_weights([256,64],'w_2'),
                            'output_w':self.init_weights([64,self.configure.n_classes],'output_w')}
            self.bias = {'b_1':self.init_bias([256],'b_1'),
                         'b_2':self.init_bias([64],'b_2'),
                         'output_b':self.init_bias([self.configure.n_classes],'output_b')}
            dense_1 = tf.add(tf.matmul(self.output,self.weights['w_1']),self.bias['b_1'],name='dense_1')
            dense_2 = tf.add(tf.matmul(dense_1,self.weights['w_2']),self.bias['b_2'],name='dense_2')
            self.preds = tf.add(tf.matmul(dense_2,self.weights['output_w']),self.bias['output_b'],name='preds')
        with tf.variable_scope('loss_and_optimizer'):
            # Loss function; Divide by batch_size to make the learning independent of batch_size
            # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
            self.loss = tf.reduce_mean(categorical_crossentropy(self.y_,self.preds))/tf.cast(tf.shape(self.x_)[0],tf.float32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_,1), tf.argmax(self.preds,1)), tf.float32))
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.configure.max_gradient_norm)
            # Optimization: global_step just keeps track of the number of batches seen so far
            self.global_step = tf.Variable(0, trainable=False)
            self.lr = self.configure.learning_rate;
            #self.lr = tf.train.exponential_decay(self.configure.learning_rate, self.global_step, self.configure.max_global_steps_assumed, 0.1)
            self.update_step = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    def init_weights(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.1),name=name)
    def init_bias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape = shape),name=name)
    def make_data_for_rnn(self, x_data, y_data):
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
# Create model template
configure = Configure();
model = rnn_tf(configure);
# Make data compatible with the architecture defined
x_train, x_test = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train, y_train = model.make_data_for_rnn(x_train, y_train)
x_test, y_test = model.make_data_for_rnn(x_test, y_test)
# Train the model
with tf.Session() as sess:
    tf.global_variables_initializer().run();
    test_every = 1;
    loss_acc_record = {}; loss_acc_record['train']=[]; loss_acc_record['test']=[];
    for epoch in range(configure.n_epochs):
        # Training
        train_loss, accuracy = 0, 0;
        for i in range(x_train.shape[0]):
            result = sess.run([model.update_step,model.loss,model.accuracy,model.preds,model.output],
                              feed_dict={model.x_:x_train[i],model.y_:y_train[i]})
            progressBar(i,x_train.shape[0],result[1],result[2])
            train_loss+=result[1];
            accuracy+=result[2];
        train_loss/=x_train.shape[0];
        accuracy/=x_train.shape[0];
        loss_acc_record['train'].append([epoch,train_loss,accuracy])
        print('\n')
        print('Epoch: {}, Training: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,train_loss,accuracy))
        progressBar(x_train.shape[0],x_train.shape[0],train_loss,accuracy)
        # Validation
        if epoch%test_every==0:
            test_loss, accuracy = 0, 0;
            for i in range(x_test.shape[0]):
                result = sess.run([model.loss,model.accuracy,model.preds,model.output],
                                  feed_dict={model.x_:x_test[i],model.y_:y_test[i]})
                progressBar(i,x_test.shape[0],result[0],result[1])
                test_loss+=result[0];
                accuracy+=result[1];
            test_loss/=x_test.shape[0];
            accuracy/=x_test.shape[0];
            loss_acc_record['test'].append([epoch,test_loss,accuracy])
            print('\n')
            print('Epoch: {}, Testing: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,test_loss,accuracy))
        # Model Saving/ Record data/ Tensorboard Writer
        # <...>
    # Stack Results
    loss_acc_record['train'] = np.stack(loss_acc_record['train'])
    loss_acc_record['test'] = np.stack(loss_acc_record['test'])
## Plot graphs
#fig, ax = plt.subplots(1,2)
#ax[0].plot(loss_acc_record['train'][:,0],loss_acc_record['train'][:,1],'b',loss_acc_record['test'][:,0],loss_acc_record['test'][:,1],'r');
#ax[0].set_xlabel('EPOCHS'); ax[0].set_ylabel('LOSS'); ax[0].set_title('TRAIN vs TEST LOSS'); ax[0].grid();
#ax[1].plot(loss_acc_record['train'][:,0],loss_acc_record['train'][:,2],'b',loss_acc_record['test'][:,0],loss_acc_record['test'][:,2],'r');
#ax[1].set_xlabel('EPOCHS'); ax[1].set_ylabel('ACCURACY'); ax[1].set_title('TRAIN vs TEST ACC'); ax[1].grid();