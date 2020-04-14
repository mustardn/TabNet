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
