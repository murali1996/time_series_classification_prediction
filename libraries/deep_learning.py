# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:38:56 2018
@author: murali.sai
"""
#%% Libraries
import datasets.data_reader
from libraries.helpers import train_test_split, progressBar

import numpy as np, os
from matplotlib import pyplot as plt
import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Reshape, Flatten, Input
#from keras.layers import Conv1D, MaxPooling1D
#from keras.layers import BatchNormalization, Dropout
#from keras.layers import LSTM, GRU, Masking
#from keras.optimizers import SGD, RMSprop, Adam

#%% PART 1
# TODO:
# 1. Implement custom objectives:
#    cross-entropy (done), cosine distance,
#    regression error, hinge loss
# 2. Implement network architectures:
#    MLP (done), RNN, CNN
# 3. For each architecture, train a classifier using
#    cross-entropy, cosine distance, and hinge loss.
#    Plot loss/accuracy as a function of epoch on one graph.
#    (9 models total)
#%% 1.0.1 Load clean data
x, y_labels = datasets.data_reader.read_clean_dataset(summary=True)
y = datasets.data_reader.one_hot(y_labels)
x_train, y_train, x_test, y_test = train_test_split(x, y)
feat_dim, out_dim = x.shape[1], y.shape[1]
#%% 1.0.2 Some Data Analysis on clean data
# Shapes
for data in [x_train, y_train, x_test, y_test]:
    print(data.shape)
# Range of values
print('max and min in x_train: {}, {}'.format(np.max(x_train),np.min(x_train)))
print('max and min in x_test: {}, {}'.format(np.max(x_test),np.min(x_test)))
print('any NaNs? {}, {}'.format(np.sum(np.isnan(x_train)),np.sum(np.isnan(x_test))))
# Plots
for label in range(y_train.shape[-1]):
    labelled = x_train[np.where(y_train[:,label]==1)[0],:]
    fig_rows, fig_cols = 2, 2
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(16, 5))
    row_inds = np.random.choice(labelled.shape[0],fig_rows*fig_cols,replace=False);
    for fig_row in range(fig_rows):
        for fig_col in range(fig_cols):
            ax[fig_row,fig_col].plot(labelled[row_inds[fig_row*fig_cols+fig_col],:]);
            ax[fig_row,fig_col].set_title('{}'.format(row_inds[fig_row*fig_cols+fig_col]));
    for axx in ax.flat: # Hide x labels and tick labels for top plots and y ticks for right plots.
        axx.label_outer()
    fig.suptitle('Label: {}'.format(label))
# PCA; Dimensionality Reduction and TSNE Visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=x.shape[-1]).fit(x);
plt.figure(); plt.plot(pca.explained_variance_); plt.grid(); plt.title('PCA Cummulative Variance across all dims')
pca_top2 = pca.transform(x)[:,:2]
plt.figure(); plt.scatter(pca_top2[:,0],pca_top2[:,1], s=0.01); plt.title('PCA Top-2 Dimensions')
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=100)
tsne_dims2 = tsne.fit_transform(x)
plt.figure(); plt.scatter(tsne_dims2[:,0],tsne_dims2[:,1], s=0.1); plt.title('TSNE (down-to) 2 Dimensions')
#%% 1.1 Write custom function
from libraries import losses
#%% 1.2.1 MLP for time-series classification
# <Implemenetation in Keras already avaialble>
class Configure_MLP(object):
    def __init__(self):
        # 1. Architecture
        self.dense_layers, self.activation = [128, 256, 64], tf.nn.relu;
        self.dropout_rate = [0.0, 0.2, 0.1];
        self.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        # 2. Training and optimization
        self.batch_size, self.n_classes = 128, 10;
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs = 10;
        self.patience = 7; # epochs until before terminating the training process
        # 3. Directories, Sub-directories, Paths
        self.main_dir = './logs';
        self.model_dir = os.path.join(self.main_dir, 'mlp_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best')
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best')
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
    def create_folders_(self):
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.model_save_training):
            os.mkdir(self.model_save_training)
        if not os.path.exists(self.model_save_inference):
            os.mkdir(self.model_save_inference)
        if not os.path.exists(self.tf_logs):
            os.mkdir(self.tf_logs)
        if not os.path.exists(self.images):
            os.mkdir(self.images)
class mlp_tf(object):
    def __init__(self):
        print('...')
#hidden = tf.layers.dense(inputs=input, units=1024, activation=tf.nn.relu)
#output = tf.layers.dense(inputs=hidden, units=labels_size)
#%% 1.2.2 RNN for Time-Series Classification
class Configure_RNN(object):
    def __init__(self):
        # 1. Architecture
        self.rnn_units, self.state_activation = [128, 128], tf.nn.tanh;
        self.keep_prob_rnn, self.drop_rate_dense = 0.8, 0.2;
        self.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
        # 2. Training and optimization
        self.batch_size, self.n_classes = 128, 10;
        self.max_gradient_norm, self.learning_rate = 5, 0.001;
        self.n_epochs = 10;
        self.patience = 7; # epochs until before terminating the training process
        # 3. Directories, Sub-directories, Paths
        self.main_dir = './logs';
        self.model_dir = os.path.join(self.main_dir, 'rnn_tf_'+self.custom_loss);
        self.model_save_training = os.path.join(self.model_dir, 'train_best')
        self.model_save_inference = os.path.join(self.model_dir, 'infer_best')
        self.tf_logs = os.path.join(self.model_dir, 'tf_logs');
        self.images = os.path.join(self.model_dir, 'images');
    def create_folders_(self):
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.model_save_training):
            os.mkdir(self.model_save_training)
        if not os.path.exists(self.model_save_inference):
            os.mkdir(self.model_save_inference)
        if not os.path.exists(self.tf_logs):
            os.mkdir(self.tf_logs)
        if not os.path.exists(self.images):
            os.mkdir(self.images)
class rnn_tf(object):
    def __init__(self, configure):
        self.configure = configure;
        with tf.variable_scope('inputs'):
            self.training = tf.placeholder(tf.bool) # True: training phase, False: testing/inference phase
            self.x_ = tf.placeholder(tf.float32, shape=[None,None,1]) # [batch_size,past_time_steps,num_fetaures]
            self.y_ = tf.placeholder(tf.float32, shape=[None,self.configure.n_classes]) # [batch_size,n_classes]
        with tf.variable_scope('multi_layer_rnn'):
            with tf.variable_scope('forward'):
                rnn_cells_forward = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, activation=self.configure.state_activation) for n in self.configure.rnn_units]
                rnn_stack_forward = tf.nn.rnn_cell.MultiRNNCell(rnn_cells_forward)
                #rnn_stack_forward = tf.contrib.rnn.DropoutWrapper(rnn_stack_forward, output_keep_prob=self.configure.keep_prob_rnn)
                outputs_forward, state_forward = tf.nn.dynamic_rnn(rnn_stack_forward, self.x_, dtype = tf.float32)
            with tf.variable_scope('backward'):
                x_backward_ = tf.reverse(self.x_, axis=[1], name='x_backward_')
                rnn_cells_backward = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, activation=self.configure.state_activation) for n in self.configure.rnn_units]
                rnn_stack_backward = tf.nn.rnn_cell.MultiRNNCell(rnn_cells_backward)
                #rnn_stack_backward = tf.contrib.rnn.DropoutWrapper(rnn_stack_backward, output_keep_prob=self.configure.keep_prob_rnn)
                outputs_backward, state_backward = tf.nn.dynamic_rnn(rnn_stack_backward, x_backward_, dtype = tf.float32)
            self.output =tf.concat([outputs_forward[:,-1,:],outputs_backward[:,-1,:]],axis=-1) # [batch_size,2*self.configure.rnn_units[-1]]
        with tf.variable_scope('dense_layers'):
            self.weights = {'w_1':self.init_weights([2*self.configure.rnn_units[-1],256],'w_1'),
                            'w_2':self.init_weights([256,64],'w_2'),
                            'output_w':self.init_weights([64,self.configure.n_classes],'output_w')}
            self.bias = {'b_1':self.init_bias([256],'b_1'),
                         'b_2':self.init_bias([64],'b_2'),
                         'output_b':self.init_bias([self.configure.n_classes],'output_b')}
            dense_1 = tf.add(tf.matmul(self.output,self.weights['w_1']),self.bias['b_1'],name='dense_1')
            dense_1 = tf.layers.dropout(dense_1, rate=self.configure.drop_rate_dense, training=self.training)
            dense_2 = tf.add(tf.matmul(dense_1,self.weights['w_2']),self.bias['b_2'],name='dense_2')
            self.preds = tf.add(tf.matmul(dense_2,self.weights['output_w']),self.bias['output_b'],name='preds')
        with tf.variable_scope('loss_and_optimizer'):
            # Loss function
            self.loss = (tf.reduce_sum(getattr(losses, self.configure.custom_loss)(self.y_,self.preds))/tf.cast(tf.shape(self.x_)[0],tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_,1), tf.argmax(self.preds,1)), tf.float32))
            # Calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.configure.max_gradient_norm)
            # Optimization and Update
            self.lr = self.configure.learning_rate;
            self.global_step = tf.Variable(0, trainable=False) # global_step just keeps track of the number of batches seen so far
            #self.lr = tf.train.exponential_decay(self.configure.learning_rate, self.global_step, self.configure.max_global_steps_assumed, 0.1)
            self.update_step = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    def init_weights(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.1),name=name)
    def init_bias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape = shape),name=name)
    def make_data_for_rnn(self, x_data, y_data): # Input SHAPES:: x_data:[n_samples,n_timesteps,n_features] y_data:[n_samples,n_classes]
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
# Set configuration and create RNN model template
configure = Configure_RNN(); configure.create_folders_();
model = rnn_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_rnn(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_rnn(x_test_, y_test)
# Training and Inference
test_every, start_epoch = 1, 0; # Define when to do testing and also set the starting epoch
if start_epoch==0: # Useful when restarting the training from an earlier stopped training phase
    train_loss_, train_loss_min, train_loss_min_epoch = [], np.inf, 0;
    train_acc_, train_acc_max, train_acc_max_epoch = [], -np.inf, 0;
    test_loss_, test_loss_min, test_loss_min_epoch = [], np.inf, 0;
    test_acc_, test_acc_max, test_acc_max_epoch = [], -np.inf, 0;
with tf.Session() as sess:
    # Initialize log writer and saver for this session and add current graph to tensorboard
    saver = tf.train.Saver() # Saving Variables and Constants
    writer = tf.summary.FileWriter(model.configure.tf_logs);  # Tensorboard
    writer.add_graph(sess.graph);
    # Initialize or load the (best) Model so far; Useful for restarting training
    if start_epoch!=0:
        saved_path = os.path.join(model.configure.model_save_inference, "model.ckpt")
        saver.restore(sess, saved_path); print("Model restored from path: {}".format(saved_path))
    else:
        tf.global_variables_initializer().run()
    # Initialize global variables and save the best models as you go training
    # Early Stopping is available with patience
    for epoch in range(model.configure.n_epochs):
        print('\n');print('EPOCH: {}'.format(epoch));
        # Training
        train_loss, accuracy = 0, 0;
        for i in range(x_train_.shape[0]):
            result = sess.run([model.update_step,model.loss,model.accuracy,model.preds,model.output],
                              feed_dict={model.training:True,model.x_:x_train_[i],model.y_:y_train_[i]})
            progressBar(i,x_train_.shape[0],result[1],result[2])
            train_loss+=result[1];
            accuracy+=result[2];
        train_loss/=x_train_.shape[0]; accuracy/=x_train_.shape[0];
        train_loss_.append(train_loss_); train_acc_.append(accuracy)
        print('\n');print('Epoch: {}, Training: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,train_loss,accuracy));
        # Validation
        if epoch%test_every==0:
            test_loss, accuracy = 0, 0;
            for i in range(x_test_.shape[0]):
                result = sess.run([model.loss,model.accuracy,model.preds,model.output],
                                  feed_dict={model.training:False,model.x_:x_test_[i],model.y_:y_test_[i]})
                progressBar(i,x_test_.shape[0],result[0],result[1])
                test_loss+=result[0];
                accuracy+=result[1];
            test_loss/=x_test_.shape[0]; accuracy/=x_test_.shape[0];
            test_loss_.append(train_loss_); test_acc_.append(accuracy)
            print('\n'); print('Epoch: {}, Testing: Avg. Loss: {:.4f} and Avg. Accuarcy: {:.4f}'.format(epoch,test_loss,accuracy))
        # Model Saving/ Record data/ Tensorboard Writer
        if True:
            # Save variables and tensors to file
            if epoch>1 and train_loss_[-1]<=train_loss_min:
                train_loss_min_epoch, train_loss_min = epoch, train_loss_[-1]
                train_save_path = saver.save(sess, os.path.join(model.configure.model_save_training, "model.ckpt"))
                print("Model (as best train) saved in path: {} after epoch: {}".format(train_save_path,epoch))
            if epoch>1 and test_loss_[-1]<=test_loss_min:
                test_loss_min_epoch, test_loss_min = epoch, test_loss_[-1];
                infer_path = saver.save(sess, os.path.join(model.configure.model_save_inference, "model.ckpt"))
                print("Model (as best val) saved in path: {} after epoch: {}".format(infer_path,epoch))
            # Patience and EarlyStopping
            if (epoch-test_loss_min_epoch)>model.configure.patience:
                break;
            # Flush the data
            print('Tensorboard logs Saved at {}'.format(writer.get_logdir()))
            writer.flush()
# Stack Results and plot graphs
train_loss_, test_loss_ = np.stack(train_loss_), np.stack(test_loss);
train_acc_, test_acc_ = np.stack(train_acc_), np.stack(test_acc_);
fig, ax = plt.subplots(1,2)
ax[0].plot(range(len(train_loss_)), train_loss_,'b',range(len(test_loss_)), test_loss_,'b');
ax[0].set_xlabel('EPOCHS'); ax[0].set_ylabel('LOSS'); ax[0].set_title('TRAIN vs TEST LOSS'); ax[0].grid();
ax[1].plot(range(len(train_acc_)), train_acc_,'b',range(len(test_acc_)), test_acc_,'b');
ax[1].set_xlabel('EPOCHS'); ax[1].set_ylabel('ACCURACY'); ax[1].set_title('TRAIN vs TEST ACC'); ax[1].grid();
# Save current python environment variables for future references
# <...>
#%% 1.2.3 CNN for Time-Series Classification


#%% PART 2
# TODO:
# 1. Classify each data point in D_corrupt.
#    Feel free to use any of the network architectures/objectives above,
#    and to perform any analysis or pre-processing on the data
#    that may improve the classification accuracy.
#    With your submission, include the results as a shape = (30000,)
#    Numpy array named corrupt_labels.npz with labels in [0,1,...,8,9].
# 2. Document the methods/strategies you used in part 2.1 in your README.
#%% 2.1 Load corrupted dataset
x, x_len = datasets.data_reader.read_corrupted_dataset(summary=True)



#%% PART 3
# TODO:
# 1. Predict the next 25 samples for each data point in D_corrupt.
#    With your submission, include the results as a shape = (30000,25)
#    Numpy array named corrupt_prediction.npz.
