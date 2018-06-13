## Author Details

Name: J Sai Muralidhar, jsaimurali001@gmail.com
Assignment: Deep Learning


**FOLDER STRUCTURE** 

In this submission, you will be finding a total of 8 code-files organized into two main folders. The folder **libraries** contain the main driver codes along with other essential helper functions. The folder **models** contain the model architecture codes along with their configuration codes written in tensorflow. Both the classification as well as sequence prediction codes are organized in this folder itself. All logs saved during training and optimization, including the configuration details are saved in the model respective folder in **logs** folder.

    .
    ├── datasets
    │   ├── figs
    │   ├── corrupted_series.npz
    │   ├── full.npz
    │   └── data_reader.py                # Code file to read clean and corrupted data
    ├── libraries
    │   ├── losses.py                     # categorical_crossentorpy, cosine_distance, regression_error, hinge_loss (with basic tf ops) 
    │   ├── helpers.py                    # train_test_split, progressBar functions available </br>
    │   ├── deep_learning.py              # driver file for the three parts described below </br>
    │   └── data_analysis.py              # data analysis on clean data; pca, tSNE, correlation, fft, etc.
    ├── models
    │	├── ts_classification
     	│	├── __init__
     	│	├── ts_mlp                    # A dense layer neural network for time-series classification
     	│	├── ts_rnn                    # A recurrent neural network for time-series classification
     	│	└── ts_cnn                    # A (1D) convolutional neural network for time-series classification
    │	├── ts_prediction
     	│	├── __init__
     	│	└── ts_seq2seq                # A recurrent sequence2sequence neural network for time-series prediction
    ├── logs
	│	├── _model_name_+_loss_function_   # configurations details, training logs, variable summaries, tensorboard, etc.
     	│	├── tf_logs                   # tensorflow Writer logs for tensorboard, variable summaries, histograms and projections
     	│	│	└── ...
     	│	├── infer_best                # Model saving during validation/inference
     	│	├── train_best                # Model saving during training
     	│	└── model_configs             # Configuration files with architectural parameter details 
            
    ├── images
    │   └── _model_name_+_loss_function_.png # Loss and Accuracy Images for different models with different loss functions

### **PART 1**

## 1. Loss functions:
Some custom loss functions have been implemented using basic tf ops.
## 2. Data Analysis:
Before developing a neural network for time-series classification, a basic data analysis was performed on the clean-data *(cl-data)*. The cl-data has 30K data samples each of length 457 time-units. Instances of null or invalid data has not been found and the data ranged between 0 and 1. The number of samples-per-class across all classes has a mean of 3000 and standard deviation of 178.2 samples, which implies that the data is not skewed. Time-series plots of randomly picked samples from each class were drawn for visual understanding; looking for any visual patterns in common among same class samples. While it was observed that a visual similarity exists, a tSNE simulation was also done with perplexity 30 and an image of the same can be found in *images* folder. It was found that the data samples formed dense clusters in tSNE plot. A PCA analysis revealed that first 83, 143 and 239 principal componenets explained a variance of 90%, 95% and 98% respectively. While techniques such as fft, #mean-crossings, signal-energy were also computed, their analysis didn't prove to be of much importance. 
## 3. Neural Networks:
A total of 3 models: *mlp*, *rnn* and *cnn* were developed in tensorflow for time-series classification task. While *mlp* and *cnn* approaches had constrains on flexible time-series lengths, the *rnn* model overcame that problem. Individual configure class files were also developed for each model so as to facilitate quick hyper-parameter and architectural modifications during training and optimization. In both *cnn* and *rnn* models, the choice to provide more than one dimenstion per time-series unit has been added i.e. simple configure file changes will reflect in architectural modifications. These models were then trained on *cl-data* using different loss functions and the details of same can be found in respective *logs* folders.

During training, it was trivial that the *rnn* model consumed a lot more time compared to other networks (457 time-units is a big number). In *rnn* network architecture, dense layers on top of recurrent bi-directional LSTM cells were used. In *cnn* network, 3, 5 and 7 sized 1D conv filters were used along with dense layers towards the end. In *mlp* network, series of dense layers were only used. Regularization at necessary layers in all three models was also incorporated. For model training, 24k samples randomly drawn from the 30k sample set was used for training and the rest 6K was kept aside for testing/inference purpose. All logs during training and the best models obtained were saved in corresponding log folders.

Analysis such as what the conv layers have learnt, or analysis on latent embeddings, etc were left for future work. The architectural parameters like number of filters, cells, units, etc. were not fully explored during this experimentation. Also, some modifications that couldn't be incorporated right now but can be analyzed as future work are:

1. Attention mechanism in RNN. Although LSTMs are designed to remember information for over long time-units, too long time-units makes remeberance of very long term memory challenging. Input-Attention mechanism can come handy in such scenarios. Truncated BPTT technique can also be used along with Attention mechanism in RNNs.
2. A fully-convolutional RNN (FC-RNN) might overcome the long training time of RNNs in time-series data, data such as the one in this experiment. While the fully-convolutional part of the architecture captures short-term patterns in the data, RNN on top of it can better model these recurring short-term patterns in lengthy time-series data.
3. An approach similar to (2) wherein we employ auto-encoders to capture short-term time series pattern for further modelling. Also, fully-convolutional architecture that acts as an auto-encoder can be developed to encode each input time-series into latent vector. A fully-convolutional because it relaxes the need for fixed length time-series data.
4. While using Deep Learning was the main objective in this experiment, other ML techniques such as k-NN can also be used for classification of the cl-data. But it again imposes a constraint on flexibilty wrt length of training and testing time-series data.
5. Short-term feature extraction; Breaking the lengthy sequence into consecutive overlapping short-segments of data and extracting suitable features for each of the short-segments, which then act as a summarized time-series information. MFCC feature extraction is a suitable example here.

### **Part 2**
## 1. Corrupted Data *(cr-data)* Classification:
It was mentioned that the *cr-data* was sampled from same distribution as the *cl-data*. Hence, prior to inference on *cr-data*, data similar to *cr-data* was generated from the 6K sample set set aside for inference purpose. Analysis of the different models on this generated data gave some results to expect on *cr-data* as well as helped to choose the importance of each model in ensemble result. Also, data analysis, as performed on *(cl-data)* was also done on the *(cr-data)*.

During inference, inputs to different models were given in different manner. While *mlp* and *cnn* models required a fixed length input (same as during training), *rnn* was fed with the effective length input. The effective length for *cr-data* samples is the legth of each corrupted sample without the trailing zeros.

Finally, an ensemble result collated from best models in *mlp*, *rnn* and *cnn* architectures was used in classification of corrupted data *(cr-data)*. While a high weightage is alloted for *rnn* model, equal weightage is alloted to the rest of the models. The results are saved in *corrupt_labels.npz* file.

### **Part 3**
## 1. Time-series Prediction:
A simple recurrent seq2seq network was developed for the task of time-series prediction. At encoder, a bi-directional LSTM multi-cell unit was utilized. The output of encoder was then passed through a dense network and then utilized as input for uni-directional multi-cell decoder. *Schedule Training* technique was incorporated during training phase. 

Due to unavailability of good GPU resources, not many experiments could be performed. For training the seq2seq model, *(cl-data)* was utilized.  The encoder side was treated with 100 time-units (picked randomly) from each sample in *cl-data*; 100 is choosen because each sample in *(cr-data)* had at-least 100 time-units. A more effective approach would be to train the encoder with lengthier time-units as well so as to capture long-term data patterns. Several un-explored ideas described in *Part 1* can also be adopted here, such as attention-mechanism, fully-convolutional layers, etc. to make the architecture more robust and effective to any length test-data. The results with this simple architecture are saved in *corrupt_prediction.npz*.

**Requirements** 
To set up the environment, please do:
```
git clone https://github.com/murali1996/ts_classification_prediction
pip install -r requirements.txt
```