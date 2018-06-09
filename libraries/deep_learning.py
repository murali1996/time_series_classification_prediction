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
# Checking the Shapes and the range of values
for data in [x_train, y_train, x_test, y_test]:
    print(data.shape)
print('max and min in x_train: {}, {}'.format(np.max(x_train),np.min(x_train)))
print('max and min in x_test: {}, {}'.format(np.max(x_test),np.min(x_test)))
print('any NaNs? {}, {}'.format(np.sum(np.isnan(x_train)),np.sum(np.isnan(x_test))))
# Number of samples per batch in x
n_samples = [np.sum(y_labels==label) for label in np.unique(y_labels)]
plt.figure(); bar_gr = plt.bar(np.unique(y_labels),n_samples); plt.xlabel('LABELS'); plt.ylabel('COUNT'); plt.title(' % Samples per class');
for rect in bar_gr:
    plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), '{:.2f}'.format(100*rect.get_height()/np.sum(n_samples)), ha='center', va='bottom')
# Randomly plot some sequences from each label [0,1,2...,9]
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
#%% 1.2.1 MLP for time-series classification using tensorflow library #<Implemenetation in Keras already provided>
from models.ts_classification.ts_mlp import Configure_MLP, MLP_tf
# Create Model/ Load Model
configure = Configure_MLP(); configure.create_folders_();
model = MLP_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = x_train.copy(), x_test.copy(); 
x_train_, y_train_ = model.make_data_for_batch_training_MLP(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_MLP(x_test_, y_test)
#%% 1.2.2 RNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_rnn import Configure_RNN, RNN_tf
# Create Model/ Load Model
configure = Configure_RNN(); configure.create_folders_();
model = RNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_RNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_RNN(x_test_, y_test)
#%% 1.2.3 CNN for Time-Series Classification using tensorflow library
from models.ts_classification.ts_cnn import Configure_CNN, CNN_tf
# Create Model/ Load Model
configure = Configure_CNN(); configure.create_folders_();
model = CNN_tf(configure);
# Make data compatible with the architecture defined; function available in model class itself
x_train_, x_test_ = np.expand_dims(x_train,axis=-1), np.expand_dims(x_test,axis=-1);
x_train_, y_train_ = model.make_data_for_batch_training_RNN(x_train_, y_train)
x_test_, y_test_ = model.make_data_for_batch_training_RNN(x_test_, y_test)
#%% 1.2.4 Train the model selected from [1.2.1, 1.2.2, 1.2.3] with one of the three loss function
model.configure.custom_loss = 'cosine_distance' # categorical_crossentropy, cosine_distance, regression_error, hinge_loss
'''
# Other parameters like batch_size, n_features, learning_rate etc. can be easily modified by calling them through model.configure.*
# For ease of implementation, all such parameters are pre-set in the respective configure files
# It is preferable to use model.configure.dense_activation as tf.nn.tanh instead of tf.nn.relu when using 'hinge_loss'
'''
# Save current configuration 
#--->
#---> Code to save config file to be added
# Training and Inference
tf.reset_default_graph()
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
#%% PART 2
# TODO:
# 1. Classify each data point in D_corrupt.
#    Feel free to use any of the network architectures/objectives above,
#    and to perform any analysis or pre-processing on the data
#    that may improve the classification accuracy.
#    With your submission, include the results as a shape = (30000,)
#    Numpy array named corrupt_labels.npz with labels in [0,1,...,8,9].
# 2. Document the methods/strategies you used in part 2.1 in your README.
#%% 1.0 Load corrupted dataset
x, x_len = datasets.data_reader.read_corrupted_dataset(summary=True)
#%% 1.1 Classify each data point
# Load RNN Model
from models.ts_classification.ts_rnn import Configure_RNN, RNN_tf
# Load configure data from path
#----> To be added #configure = configure_RNN();
# Initialize model
model = RNN_tf(configure);
# Inference copy code 

#for each prediction, take argmax and report it

#%% PART 3
# TODO:
# 1. Predict the next 25 samples for each data point in D_corrupt.
#    With your submission, include the results as a shape = (30000,25)
#    Numpy array named corrupt_prediction.npz.
'''
A generic model has been developed for all classes to predict the next 25 samples. A more effective way would be to train individual model per class
Seq2Seq Learning with scheduled training has been used in prediction
'''
# 3.0.1 Data Preperation for model training