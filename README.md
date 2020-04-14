# TabNet

## Model Architectures

**Inputs:** An array of mel-scaled spectrograms for each audio frame.

**Outputs:** For each spectrogram frame, a 6x21 matrix of probabilities is output, where each row represents a string, and each column represents a fret number on the fretboard. Index 0 of the array represents fret 0 (string being played open), index 1 represents fret 1, index 2 fret 2, and so on.

### Convolutional Neural Network Diagram:

The below architecture is adapted from TabCNN [1]. The model consists of 3 convolutional layers using relu activations with (2,2) filters, a max pooling layer, and two dense layers. A 6 dimensional softmax is applied and a distribution of probabilities are ouput for each string. 

![](/CNN_diagram.PNG)


### Convolutional Recurrent Neural Network Diagram:

The below architecture is my proposal to improve the results of the CNN model. The architecture is very similar to the above model, however it incorporates a RNN layer. Later, I will utilize hyperparameter tuning to determine what type of RNN to use - LSTM, GRU, or bidirectional LSTM.

![](/CRNN_diagram.PNG)


## References

[1] Andrew Wiggins, Youngmoo Kim. “Guitar Tablature Estimation
with a Convolutional Neural Network”, 20th International Society
for Music Information Retrieval Conference, Delft, The Netherlands,
2019.

[2] Qingyang Xi, Rachel Bittner, Johan Pauwels, Xuzhou Ye, Juan Bello. “GuitarSet: A Dataset for Guitar Transcription”, 19th International Society for Music Information Retrieval Conference, Paris, France, 2018.

[3]  Keunwoo Choi, Gyorgy Fazekas, Kyunghyun Cho, and Mark Sandler, “A tutorial on deep learning for music information retrieval,” arXiv preprint arXiv:1709.04396, 2017.

## Code References

`animations.py` was adapted from [fretboardgtr](https://github.com/antscloud/fretboardgtr/tree/master/fretboardgtr)

`TabNet.py` was adapted from [TabCNN.py](https://github.com/andywiggins/tab-cnn/blob/master/model/TabCNN.py). 

The data was generated using [DataGenerator.py](https://github.com/andywiggins/tab-cnn/blob/master/model/DataGenerator.py). Slight modifications were made to update this code to Python 3 and Tensorflow v2.

The various metrics calculated on the test data came from [Metrics.py](https://github.com/andywiggins/tab-cnn/blob/master/model/Metrics.py). Slight modifications were made to update this code to Python 3 and Tensorflow v2.
