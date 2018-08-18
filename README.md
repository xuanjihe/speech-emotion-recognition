# speech-emotion-recognition
TensorFlow implementation of Convolutional Recurrent Neural Networks for speech emotion recognition (SER) on the IEMOCAP database.In order to address the problem of the uncertainty of frame emotional labels, we perform three pooling strategies(max-pooling, mean-pooling and attention-based weighted-pooling) to produce utterance-level features for SER.
These codes have only been tested on ubuntu 16.04(x64), python2.7, cuda-8.0, cudnn-6.0 with a GTX-1080 GPU.To run these codes on your computer, you need install the following dependency:
１、tensorflow 1.3.0
２、python_speech_features
３、wave
４、cPickle
５、numpy
６、sklearn
７、os

The detailed information of this code can be found in 3-D.pdf, you can download if from the top of this page. 

Citation
If you used this code, please kindly consider citing the following paper:

Chen, Mingyi & He, Xuanji & Yang, Jing & Zhang, Han. (2018). 3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition. IEEE Signal Processing Letters. 1-1. 10.1109/LSP.2018.2860246. 
