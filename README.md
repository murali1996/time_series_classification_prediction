## Author Details
Name: J Sai Muralidhar, jsaimurali001@gmail.com </br>
Assignment: Deep Learning </br>

**FOLDER STRUCTURE** </br>
In this submission, you will be finding a total of 9 code-files organized into two main folders. The folder **libraries** contain the main driver codes along with other essential helper functions. The folder **models** contain the model architecture codes along with their configuration codes written in tensorflow. Both the classification as well as sequence prediction codes are organized in this folder itself. All logs saved during training and optimization, including the configuration details are saved in the model respective folder in **logs** folder. </br>
. </br>
├── datasets </br>                            
|   ├── figs </br>
|   ├── corrupted_series.npz </br>
|   ├── full.npz </br>
|   └── data_reader.py                # Code file to read clean and corrupted data  </br>
├── libraries </br>
|   ├── losses.py                    # categorical_crossentorpy, cosine_distance, regression_error, hinge_loss (with basic tf ops) </br>
|   ├── helpers.py                    # train_test_split, progressBar functions available </br>
|   ├── deep_learning.py              # driver file for the three parts described below </br>
|   └── data_analysis.py              # data analysis on clean data; pca, tSNE, correlation, fft, etc. </br>
├── models </br>
|   ├── ts_classification </br>
|       ├── ts_mlp                    # A dense layer neural network for time-series classification </br>
|       ├── ts_rnn                    # A recurrent neural network for time-series classification </br>
|       └── ts_cnn                    # A (1D) convolutional neural network for time-series classification </br>
|   └── ts_prediction </br>
|       └── ts_seq2seq                # A recurrent sequence2sequence neural network for time-series prediction </br>
├── logs </br>
|   └── <model_name_+_loss_function>  # configurations details, training logs, variable summaries, tensorboard, etc. </br>
|       ├── tf_logs </br>
|       ├── infer_best </br>
|       ├── train_best </br>
|       └── model_configs </br>
└── images                            # Loss and Accuracy Images for different models with different loss functions </br>
|   └── <model_name_+_loss_function>.png </br>

**PART 1** 
1. Loss functions:
Some custom loss functions have been implemented using basic tf ops.
2. Data Analysis:
Before developing a neural network for time-series classification, a basic data analysis was performed on the clean-data *(cl-data)*. The cl-data has 30K data samples each of length 457 time-units. Instances of null or invalid data has not been found and the data ranged between 0 and 1. The number of samples-per-class across all classes has a mean of 3000 and standard deviation of 178.2 samples, which implies that the data is not skewed. Time-series plots of randomly picked samples from each class were drawn for visual understanding; looking for any visual patterns in common among same class samples. While it was observed that a visual similarity exists, a tSNE simulation was also done with perplexity 30 and an image of the same can be found in *images* folder. It was found that the data samples formed dense clusters in tSNE plot. A PCA analysis revealed that first 83, 143 and 239 principal componenets explained a variance of 90%, 95% and 98% respectively. While techniques such as fft, #mean-crossings, signal-energy were also computed, their analysis didn't prove to be of much importance. 
3. Neural Networks
A total of 3 models: *mlp*, *rnn* and *cnn* were developed in tensorflow for time-series classification task. While *mlp* and *cnn* approaches had constrains on flexible time-series lengths, the *rnn* model overcame that problem. Individual configure class files were also developed for each model so as to facilitate quick hyper-parameter and architectural modifications during training and optimization. In both *cnn* and *rnn* models, the choice to provide more than one dimenstion per time-series unit has been added i.e. simple configure file changes will reflect in architectural modifications. These models were then trained on *cl-data* using different loss functions and the details of same can be found in respective *logs* folders. 
During training, it was trivial that an *rnn* model consumed a lot more time compared to other networks (457 time-units is a big number). In *rnn* network architecture, dense layers on top of recurrent bi-directional LSTM cells were used. In *cnn* network, 3, 5 and 7 sized 1D conv filters were used along with dense layers towards the end. In *mlp* network, series of dense layers were only used. Regularization at necessary layers in all three models was also incorporated.   
Analysis such as what the conv layers have learnt, or analysis on latent embeddings, etc were left for future work. The architectural parameters like number of filters, cells, units, etc. were not fully explored during this experimentation. Also, some modifications that couldn't be incorporated right now but can be analyzed as future work are:
    (i) Attention mechanism in RNN. Although LSTMs are designed to remember information for over long time-units, too long time-units makes remeberance of very long term memory challenging. Input-Attention mechanism can come handy in such scenarios. Truncated BPTT technique can also be used along with Attention mechanism in RNNs.
    (ii) A fully-convolutional RNN (FC-RNN) might overcome the long training time of RNNs in time-series data, data such as the one in this experiment. While the fully-convolutional part of the architecture captures short-term patterns in the data, RNN on top of it can better model these recurring short-term patterns in lengthy time-series data.
    (iii) An approach similar to (ii) wherein we employ auto-encoders to capture short-term time series pattern for further modelling. Also, fully-convolutional architecture that acts as an auto-encoder can be developed to encode each input time-series into latent vector. A fully-convolutional because it relaxes the need for fixed length time-series data.
    (iv) While using Deep Learning was the main objective in this experiment, other ML techniques such as k-NN can also be used for classification of the cl-data. But it again imposes a constraint on flexibilty wrt length of training and testing time-series data.
    (v) Short-term feature extraction; Breaking the lengthy sequence into consecutive overlapping short-segments of data and extracting suitable features for each of the short-segments, which then act as a summarized time-series information. MFCC feature extraction is a suitable example here.
    
**Part 2**
1. Corrupted Data Classification:
An ensemble result collated from best models with *mlp*, *rnn* and *cnn* architectures is used in classification of corrupted data. While a high weightage is alloted for *rnn* model, equal weightage is alloted to the rest of the models. The results are saved in *corrupt_labels.npz* file.

**Part 3**
1. Time-series Prediction
A simple recurrent seq2seq network was developed for the task of time-series prediction. At encoder, a bi-directional LSTM multi-cell unit was utilized. The output of encoder was passed through a dense network and then utilized as input for uni-directional multi-cell decoder. *Schedule Training* technique was incorporated during training phase. Due to unavailability of good GPU resources, not many experiments could be performed. The encoder side was treated with 100 time-units from each sample in *cl-data* (100 is choosen because each sample in corrupted data had at-least 100 time-units). A better approach would be to train the encoder with lengthier time-units as well so as to capture long data patterns. All un-explored ideas described in *part 1* can also be adopted here, such as attention-mechanism, fully-convolutional layers, etc. to make the architecture more robust and effective to any length test-data.
The results with this simple architecture are saved in *corrupt_prediction.npz*.
