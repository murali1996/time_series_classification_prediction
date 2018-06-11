# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:38:56 2018
@author: murali.sai
"""
import datasets.data_reader
from libraries.helpers import train_test_split, progressBarSimple
import numpy as np
from matplotlib import pyplot as plt

if __name__=="__main__":
    #%% Load clean data and do some data analysis
    x, y_labels = datasets.data_reader.read_clean_dataset(summary=True)
    y = datasets.data_reader.one_hot(y_labels)
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    #%% Load corrupted dataset
    x_c, x_c_len = datasets.data_reader.read_corrupted_dataset(summary=True)
    #%% 1. Checking the Shapes and the range of values
    assert(x.shape[0]==y.shape[0])
    for data in [x_train, y_train, x_test, y_test]:
        print(data.shape)
    print('max and min in x_train: {}, {}'.format(np.max(x_train),np.min(x_train)))
    print('max and min in x_test: {}, {}'.format(np.max(x_test),np.min(x_test)))
    print('any NaNs? {}, {}'.format(np.sum(np.isnan(x_train)),np.sum(np.isnan(x_test))))

    #%% 2. Number of samples per batch in x
    n_samples = [np.sum(y_labels==label) for label in np.unique(y_labels)]
    plt.figure(); bar_gr = plt.bar(np.unique(y_labels)/np.sum(n_samples),n_samples); plt.xlabel('LABELS'); plt.ylabel('COUNT'); plt.title(' % Samples per class');
    for rect in bar_gr:
        plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), '{:.2f}%'.format(100*rect.get_height()), ha='center', va='bottom')

    #%% 3. Randomly plot some sequences from each label [0,1,2...,9]
    for label in range(y.shape[-1]):
        labelled = x[np.where(y[:,label]==1)[0],:]
        fig_rows, fig_cols = 5, 2
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(16, 5))
        row_inds = np.random.choice(labelled.shape[0],fig_rows*fig_cols,replace=False);
        for fig_row in range(fig_rows):
            for fig_col in range(fig_cols):
                ax[fig_row,fig_col].plot(labelled[row_inds[fig_row*fig_cols+fig_col],:]);
                ax[fig_row,fig_col].set_title('{}'.format(row_inds[fig_row*fig_cols+fig_col]));
        for axx in ax.flat: # Hide x labels and tick labels for top plots and y ticks for right plots.
            axx.label_outer()
        fig.suptitle('Time-Series-Plots (Samples randomly selected from) @Class_Label:: {}'.format(label))

    #%% 4. Find (within-class) Pearson Correlation for all classes; without and with time_shifts
    for label in range(y.shape[-1]):
        labelled = x[np.where(y[:,label]==1)[0],:]
        means_ = np.mean(labelled,axis=1).reshape([labelled.shape[0],1]);
        labelled = labelled-means_; labelled_T = labelled.T;
        sigma_x = np.sqrt(np.sum(labelled*labelled, axis=1)).reshape([labelled.shape[0],1]); sigma_y = sigma_x.T;
        corr_mat = (np.dot(labelled,labelled_T)+1e-4)/(np.dot(sigma_x,sigma_y)+1e-4);
        triu_indices = np.triu_indices(n=corr_mat.shape[0], m=corr_mat.shape[1], k=1) # Symmetrical and diagnols are 1
        hist, bin_edges = np.histogram(corr_mat[triu_indices],bins=10); bin_centers = (bin_edges[:-1]+bin_edges[1:])/2;
        plt.figure(); bar_gr = plt.bar( x=bin_centers, height=hist/np.sum(hist), width=0.1); plt.xticks(bin_centers);
        plt.xlabel('correlation'); plt.ylabel('hist_count'); plt.title('Within-Class Correlation Histogram; time_shift=0; Label:{}'.format(label));
        for rect in bar_gr:
            plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), '{:.3f}%'.format(100*rect.get_height()), ha='center', va='bottom')
    for label in range(y.shape[-1]): #Time and memory consuming!
        labelled = x[np.where(y[:,label]==1)[0],:]
        means_ = np.mean(labelled,axis=1).reshape([labelled.shape[0],1]);
        labelled = labelled-means_; labelled_T = labelled.T;
        corr_mat = np.array([[-np.inf]*labelled.shape[0]]*labelled.shape[0]);
        for time_shift in range(100): # Shift all rows of labelled_T down and make first column 0
            progressBarSimple(time_shift,100);
            labelled_T[1:,:] = labelled_T[0:-1,:]; labelled_T[0,:]*=0;
            sigma_x = np.sqrt(np.sum(labelled_T.T*labelled_T.T, axis=1)).reshape([labelled.shape[0],1]);
            sigma_y = sigma_x.T;
            corr_mat_temp = (np.dot(labelled,labelled_T)+1e-4)/(np.dot(sigma_x,sigma_y)+1e-4);
            corr_mat = np.maximum(corr_mat, corr_mat_temp);
        triu_indices = np.triu_indices(n=corr_mat.shape[0], m=corr_mat.shape[1], k=1) # Symmetrical and diagnols are 1
        hist, bin_edges = np.histogram(corr_mat[triu_indices],bins=10); bin_centers = (bin_edges[:-1]+bin_edges[1:])/2;
        plt.figure(); bar_gr = plt.bar( x=bin_centers, height=hist/np.sum(hist), width=0.1); plt.xticks(bin_centers);
        plt.xlabel('correlation'); plt.ylabel('hist_count'); plt.title('Within-Class Correlation Histogram; time_shiftbest_corr0; Label:{}'.format(label));
        for rect in bar_gr:
            plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), '{:.3f}%'.format(100*rect.get_height()), ha='center', va='bottom')

    #%% 5. PCA; Dimensionality Reduction and TSNE Visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x.shape[-1]).fit(x);
    plt.figure(); plt.plot(pca.explained_variance_); plt.grid(); plt.title('PCA Cummulative Variance across all dims')
    pca_top2 = pca.transform(x)[:,:2]
    plt.figure(); plt.scatter(pca_top2[:,0],pca_top2[:,1], s=0.01); plt.title('PCA Top-2 Dimensions')
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=100)
    tsne_dims2 = tsne.fit_transform(x)
    tsne_c = TSNE(n_components=2, perplexity=30, learning_rate=100)
    tsne_dims2_c = tsne_c.fit_transform(x_c)
    plt.figure();
    plt.scatter(tsne_dims2[:,0],tsne_dims2[:,1], s=0.1, c='r'); #plt.title('Clean_Data: TSNE (down-to) 2 Dimensions');
    plt.scatter(tsne_dims2[:,0],tsne_dims2[:,1], s=0.1, c='b'); #plt.title('Corrupt_data: TSNE (down-to) 2 Dimensions');
    plt.show();

    #%% 6 FFT on the time-series data
    fft_x = np.abs(np.fft.fft(x,axis=1)); fft_x = fft_x[:,:int(np.ceil(x.shape[1]/2))+1]
    for label in range(y.shape[-1]):
        labelled = fft_x[np.where(y[:,label]==1)[0],:]
        fig_rows, fig_cols = 5, 2
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(16, 5))
        row_inds = np.random.choice(labelled.shape[0],fig_rows*fig_cols,replace=False);
        for fig_row in range(fig_rows):
            for fig_col in range(fig_cols):
                ax[fig_row,fig_col].plot(labelled[row_inds[fig_row*fig_cols+fig_col],:]);
                ax[fig_row,fig_col].set_title('{}'.format(row_inds[fig_row*fig_cols+fig_col]),y=.93);
        for axx in ax.flat: # Hide x labels and tick labels for top plots and y ticks for right plots.
            axx.label_outer()
        fig.suptitle('Time-Series-FFT-Plots (Samples randomly selected from) @Class_Label:: {}'.format(label))

    #%% 7.1 Mean-Crossing Variations in each class
    a = x.copy()-np.mean(x,axis=1).reshape([x.shape[0],1]);
    b = a[:,1:]*a[:,0:-1];
    mean_crossings = np.sum(b>0,axis=1); del a,b;
    fig_rows, fig_cols = 5, 2
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(12, 6))
    for i, ax in enumerate(ax.flat):
        lab = mean_crossings[np.where(y_train[:,i]==1)[0]];
        hist, bin_edges = np.histogram(lab,bins=20); bin_centers = (bin_edges[:-1]+bin_edges[1:])/2;
        bar_gr = ax.bar( x=bin_centers, height=hist/np.sum(hist), width=2.5);
        ax.set_title('Label:{}, Mean:{:.2f}, Var:{:.2f}'.format(i,np.mean(lab),np.var(lab)),size=6,y=.93);
        ax.label_outer();
        #for rect in bar_gr:
        #    ax.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), '{:.1f}'.format(100*rect.get_height()), ha='center', va='bottom')
    fig.suptitle('Mean-Crossings');
    fig.text(0.5, 0.04, 'mean-crossing magnitudes', ha='center')
    fig.text(0.04, 0.5, 'hist_counts', va='center', rotation='vertical')
    #%% 7.2 Energy Variations in each class
    enr_ = np.sqrt(np.sum(x*x,axis=1)).reshape([x.shape[0],1]);
    fig_rows, fig_cols = 5, 2
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(12, 6))
    for i, ax in enumerate(ax.flat):
        lab = enr_[np.where(y_train[:,i]==1)[0]];
        hist, bin_edges = np.histogram(lab,bins=25); bin_centers = (bin_edges[:-1]+bin_edges[1:])/2;
        bar_gr = ax.bar( x=bin_centers, height=hist/np.sum(hist), width=0.25);  #ax.set_xticks(bin_centers);
        ax.set_title('Label:{}, Mean:{:.2f}, Var:{:.2f}'.format(i,np.mean(lab),np.var(lab)),size=7);
        ax.label_outer();
    fig.suptitle('energy');
    fig.text(0.5, 0.04, 'Samples belonging to each label (in no particular order)', ha='center')
    fig.text(0.04, 0.5, 'energy magnitudes', va='center', rotation='vertical')