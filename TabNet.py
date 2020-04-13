''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''

from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Permute, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D, Lambda, LSTM, GRU, Bidirectional
from tensorflow.keras import backend as K
from DataGenerator import DataGenerator
from sklearn.model_selection import ParameterGrid, KFold, train_test_split
from statistics import mean
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from Metrics import *


class TabNet:

    def __init__(self,
                 save_path,
                 batch_size=32,
                 epochs=8,
                 con_win_size=9,
                 spec_repr="m",
                 data_path="../data/spec_repr/",
                 id_file="id.csv"):

        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.load_IDs()

        #self.save_folder = self.save_path + self.spec_repr + " " + datetime.datetime.now().strftime(
            #"%Y-%m-%d %H:%M:%S") + "/"

        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []

        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
            self.X_dim = (self.batch_size, 192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
            self.X_dim = (self.batch_size, 128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
            self.X_dim = (self.batch_size, 320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)
            self.X_dim = (self.batch_size, 1025, self.con_win_size, 1)

        self.num_classes = 21
        self.num_strings = 6
        self.label_dim = (6, 21)

        self.y_dim = (self.batch_size, self.label_dim[0], self.label_dim[1])

    def get_saved_folder(self):

        self.save_folder = self.save_path + self.model_name + "/"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.log_file = self.save_folder + "log.txt"

    def load_IDs(self):
        csv_file = self.data_path + self.id_file
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])

    def partition_data(self, data_split=0.2):

        self.X_train, self.X_test = train_test_split(self.list_IDs, test_size=data_split, random_state=42)
        self.X_train, self.X_val = train_test_split(self.X_train, test_size=data_split, random_state=42)

        self.training_generator = DataGenerator(self.X_train,
                                                data_path=self.data_path,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                spec_repr=self.spec_repr,
                                                con_win_size=self.con_win_size)

        self.testing_generator = DataGenerator(self.X_test,
                                               data_path=self.data_path,
                                               batch_size=len(self.X_test),
                                               shuffle=False,
                                               spec_repr=self.spec_repr,
                                               con_win_size=self.con_win_size)

        self.validation_generator = DataGenerator(self.X_val,
                                                  data_path=self.data_path,
                                                  batch_size=self.batch_size,
                                                  shuffle=False,
                                                  spec_repr=self.spec_repr,
                                                  con_win_size=self.con_win_size)

    def log_model(self):
        with open(self.log_file, 'w') as fh:
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
        return K.concatenate(string_sm, axis=1)

    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:, i, :], output[:, i, :])
        return loss

    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def baseline_cnn_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=self.catcross_by_string,
                      optimizer=opt,
                      metrics=[self.avg_acc])
        self.model = model
        self.model_name = 'cnn-model'

    def cnn_model(self, param_grid):

        self.param_grid = param_grid

        # define hyperparameters
        lr = self.param_grid['lr']
        optimizer = self.param_grid['optimizer']
        c_layers = self.param_grid['c_layers']
        filters = self.param_grid['filters']
        kernel_size = self.param_grid['kernel']
        dense_size = self.param_grid['dense_size']

        try:
            del model
        except:
            pass
        K.clear_session()

        model = Sequential()
        model.add(Conv2D(32, kernel_size=kernel_size,
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())

        for i in range(c_layers):
            model.add(Conv2D(filters, kernel_size, activation='relu'))
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        if optimizer == 'Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'Nadam':
            opt = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif optimizer == 'Adadelta':
            opt = tf.keras.optimizers.Nadam(learning_rate=lr)

        model.compile(loss=self.catcross_by_string,
                      optimizer=opt,
                      metrics=[self.avg_acc])
        self.model = model
        self.model_name = 'cnn-model'

    def baseline_crnn_model(self, lr=0.001):

        # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
        # we would rather have it the other way around for LSTMs

        model = Sequential()
        model.add(Input(self.input_shape, name='input'))
        model.add(Permute((2, 1, 3)))
        model.add(Conv2D(32, kernel_size=3,
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(1, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'))
        model.add(LSTM(96))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=self.catcross_by_string,
                      optimizer=opt,
                      metrics=[self.avg_acc])

        self.model = model
        self.model_name = 'crnn-model'

    def crnn_model(self, param_grid):

        # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
        # we would rather have it the other way around for LSTMs

        self.param_grid = param_grid
        lr = self.param_grid['lr']
        optimizer = self.param_grid['optimizer']
        c_layers = self.param_grid['c_layers']
        rnn_type = self.param_grid['rnn_type']
        rnn_size = self.param_grid['rnn_size']

        try:
            del model
        except:
            pass
        K.clear_session()

        model = Sequential()
        model.add(Input(self.input_shape, name='input'))
        model.add(Permute((2, 1, 3)))
        model.add(Conv2D(32, kernel_size=3,
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())

        for i in range(c_layers):
            model.add(Conv2D(64, 3, activation='relu'))
            model.add(BatchNormalization())

        model.add(Conv2D(1, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim'))
        if rnn_type == 'LSTM':
            model.add(LSTM(rnn_size))
        elif rnn_type == 'GRU':
            model.add(GRU(rnn_size))
        elif rnn_type == 'bidirectional':
            model.add(Bidirectional(LSTM(rnn_size)))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        if optimizer == 'Adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'Nadam':
            opt = tf.keras.optimizers.Nadam(learning_rate=lr)
        elif optimizer == 'Adadelta':
            opt = tf.keras.optimizers.Adadelta(learning_rate=lr)

        model.compile(loss=self.catcross_by_string,
                      optimizer=opt,
                      metrics=[self.avg_acc])

        self.model = model
        self.model_name = 'crnn-model'

    def train(self):

        self.X_val, self.y_val = self.validation_generator[0]

        self.history = self.model.fit(x=self.training_generator,
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=(self.X_val, self.y_val),
                                      use_multiprocessing=False,
                                      workers=1)

        df = pd.DataFrame.from_dict(self.history.history)
        df.to_csv(self.save_folder + "model_history.csv")

    def show_curve(self):

        plt.plot(self.history.history['avg_acc'])
        plt.plot(self.history.history['val_avg_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def save_weights(self):
        self.model.save_weights(self.save_folder + "weights.h5")

    def test(self):
        self.X_test, self.y_gt = self.testing_generator[0]
        self.y_pred = self.model.predict(self.X_test, verbose=1)

    def save_predictions(self):
        np.savez(self.save_folder + "predictions.npz", y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):

        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(self.save_folder + "results.csv")

class HypTune(TabNet):

    def __init__(self,
                 save_path,
                 batch_size=128,
                 epochs=2,
                 con_win_size=9,
                 spec_repr="m",
                 data_path="../data/spec_repr/",
                 id_file="id.csv"):

        super().__init__(save_path,
                         batch_size=128,
                         epochs=2,
                         con_win_size=9,
                         spec_repr="m",
                         data_path="../data/spec_repr/",
                         id_file="id.csv")

    def cv_partition_data(self, train_IDs, validation_IDs, test_IDs):

        self.partition = {}
        self.partition["training"] = train_IDs
        self.partition["testing"] = test_IDs
        self.partition["validation"] = validation_IDs

        self.testing_generator = DataGenerator(self.partition['testing'],
                                                           data_path=self.data_path,
                                                           batch_size=len(test_IDs),
                                                           shuffle=False,
                                                           spec_repr=self.spec_repr,
                                                           con_win_size=self.con_win_size)

        self.training_generator = DataGenerator(self.partition['training'],
                                                data_path=self.data_path,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                spec_repr=self.spec_repr,
                                                con_win_size=self.con_win_size)

        self.validation_generator = DataGenerator(self.partition['validation'],
                                                              data_path=self.data_path,
                                                              batch_size=self.batch_size,
                                                              shuffle=False,
                                                              spec_repr=self.spec_repr,
                                                              con_win_size=self.con_win_size)

    def hyperparameter_tuning(self, model_type, **kwargs):

        grid = ParameterGrid(kwargs)
        self.model_type = model_type

        models_dict = {}
        for i, param in enumerate(grid):
            print('\nTraining model:', i)
            print('Model Parameters:', param)

            if model_type == "cnn":
                self.cnn_model(param)
                self.model_name = 'cnn-model' + str(i)
            elif model_type == "crnn":
                self.crnn_model(param)
                self.model_name = 'crnn-model' + str(i)

            self.get_saved_folder()
            self.log_model()
            self.train()
            self.save_weights()

            models_dict[i] = {}
            models_dict[i]['performance'] = self.history.history
            models_dict[i]['parameters'] = param

        self.models_dict = models_dict

    def hp_curve(self, metric, *dataframes):

        # allow function to pass DataFrame instead of using models_dict
        if dataframes:
            for df in dataframes:
                if metric=='avg_acc':
                    plt.plot(df['performance']['avg_acc'])
                    plot_title = 'Plot of Model Training Accuracies'
                elif metric=='val_avg_acc':
                    plt.plot(df['performance']['val_avg_acc'])
                    plot_title = 'Plot of Model Validation Accuracies'
        else:
            for k, v in self.models_dict.items():
                if metric=='avg_acc':
                    plt.plot(v['performance']['avg_acc'])
                    plot_title = 'Plot of Model Training Accuracies'
                elif metric=='val_avg_acc':
                    plt.plot(v['performance']['val_avg_acc'])
                    plot_title = 'Plot of Model Validation Accuracies'

        plt.title(plot_title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(self.models_dict.keys(), loc='upper left')
        plt.show()

    def SimpleNestedCV(self, n1=5, **kwargs):

        kf1 = KFold(n_splits=n1)
        ncv_models_dict = {}

        for j, (train_index1, test_index) in enumerate(kf1.split(self.list_IDs)):
            print('Fold', j)
            train_IDs = np.array(self.list_IDs)[train_index1]
            test_IDs = np.array(self.list_IDs)[test_index]
            best_fold_acc = []
            best_acc = 0
            print('Nested Fold')
            train_IDs2, validation_IDs = train_test_split(train_IDs, test_size=0.2, random_state=42)
            self.cv_partition_data(train_IDs2, validation_IDs, test_IDs)
            self.hyperparameter_tuning(**kwargs)

            for k, v in self.models_dict.items():
                # take validation accuracy from the last epoch
                acc = v['performance']['val_avg_acc'][-1]
                if acc > best_acc:
                    best_acc = acc
                    # dictionary of parameters
                    best_parameters = v['parameters']
                best_fold_acc.append(best_acc)

            print('\ntrain on best hyperparameters')
            self.fold_acc = mean(best_fold_acc)
            self.cv_partition_data(train_IDs, validation_IDs, test_IDs)
            if self.model_type == 'cnn':
                self.cnn_model(best_parameters)
                self.model_name = 'cnn-model-fold' + str(j)
            elif self.model_type == 'crnn':
                self.crnn_model(best_parameters)
                self.model_name = 'crnn-model-fold' + str(j)
            self.train()
            print('')
            ncv_models_dict[j] = {}
            ncv_models_dict[j]['performance'] = self.history.history
            ncv_models_dict[j]['parameters'] = best_parameters

        self.ncv_models_dict = ncv_models_dict

    def NestedCV(self, n1=5, n2=5, **kwargs):

        kf1 = KFold(n_splits=n1)
        kf2 = KFold(n_splits=n2, shuffle=True, random_state=1)
        ncv_models_dict = {}

        for j, (train_index1, test_index) in enumerate(kf1.split(self.list_IDs)):
            print('outer fold', j)
            train_IDs = np.array(self.list_IDs)[train_index1]
            best_fold_acc = []
            best_acc = 0
            for i, (train_index2, validation_index) in enumerate(kf2.split(train_IDs)):
                print('nested fold', i)
                train_IDs2 = train_IDs[train_index2]
                validation_IDs = train_IDs[validation_index]
                self.cv_partition_data(train_IDs=train_IDs2, validation_IDs=validation_IDs)
                self.hyperparameter_tuning(**kwargs)

                for k, v in self.models_dict.items():
                    # take validation accuracy from the last epoch
                    acc = v['performance']['val_avg_acc'][-1]
                    if acc > best_acc:
                        best_acc = acc
                        # dictionary of parameters
                        best_parameters = v['parameters']
                    best_fold_acc.append(best_acc)

            self.fold_acc = mean(best_fold_acc)
            test_IDs = np.array(self.list_IDs)[test_index]
            self.cv_partition_data(train_IDs=train_IDs, test_IDs=test_IDs)
            print('train on best hyperparameters')
            if self.model_type == 'cnn':
                self.cnn_model(best_parameters)
                self.model_name = 'cnn-model-fold' + str(j)
            elif self.model_type == 'crnn':
                self.crnn_model(best_parameters)
                self.model_name = 'crnn-model-fold' + str(j)
            self.train()
            ncv_models_dict[j] = {}
            ncv_models_dict[j]['performance'] = self.history.history
            ncv_models_dict[j]['parameters'] = best_parameters

        self.ncv_models_dict
