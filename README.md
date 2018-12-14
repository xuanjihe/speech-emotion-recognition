# speech-emotion-recognition
TensorFlow implementation of Convolutional Recurrent Neural Networks for speech emotion recognition (SER) on the IEMOCAP database.In order to address the problem of the uncertainty of frame emotional labels, we perform three pooling strategies(max-pooling, mean-pooling and attention-based weighted-pooling) to produce utterance-level features for SER.
These codes have only been tested on ubuntu 16.04(x64), python2.7, cuda-8.0, cudnn-6.0 with a GTX-1080 GPU.To run these codes on your computer, you need install the following dependency:    
* tensorflow 1.3.0
* python_speech_features
* wave
* cPickle
* numpy
* sklearn
* os

## Demo
For running a demo, after forking the repository, run the following scrit:

          python zscore.py
   
          python ExtractMel.py
   
          python model.py
   
## Note
There are some little differences betweem the implementation and the paper, eg.when run python model.py, you will find that the recognition rate of happy is very poor, which is caused by the imbalance of the training samples. An effective method is to use the happy sample twice.

If you want to download the `IEMOCAP` database, you can access the link:https://sail.usc.edu/iemocap/release_form.php

The detailed information of this code can be found in 3-D.pdf, you can download if from the top of this page. 
## Author
Xuanji He<br> 
E-mail: xuanjihe@163.com

## Citation
If you used this code, please kindly consider citing the following paper:

Mingyi Chen, Xuanji He, Jing Yang, Han Zhang, "3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition", IEEE Signal Processing Letters, vol. 25, no. 10, pp. 1440-1444, 2018.
