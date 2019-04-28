import autograd.numpy as np
from autograd import grad
import string
import math
import os
import json
import dill
from keras.utils import np_utils


DEFAULT_ALPHABET = string.printable


def load_saved_model(model_path):
    with open(model_path, 'rb') as f:
        return dill.load(f)


class SimpleRecurrentModel(object):
    def __init__(self, window_size, alphabet, window_shift=0, preprocess=id):
        self.num_chars = len(alphabet)
        self.window_size = window_size
        self.alphabet = {alphabet[i]: i for i in range(len(alphabet))}
        self.window_shift = window_shift
        self.preprocess = preprocess
        self.weights = np.zeros(shape=(self.num_chars * self.window_size + 2), dtype='float32')

    def train(self, train_inputs, train_labels, batch_size, learning_rate, steps):
        train_inputs = [self.preprocess(i) for i in train_inputs]
        train_inputs, train_labels = self.assemble_data(train_inputs, train_labels)
        self._raw_train(train_inputs, train_labels, batch_size, learning_rate, steps)

    def _raw_train(self, train_inputs, train_labels, batch_size, learning_rate, steps):
        def loss(params, sample_inputs, sample_labels):
            output = SimpleRecurrentModel._raw_inference(params, sample_inputs)
            return -np.sum(output * sample_labels + (1 - output) * (1 - sample_labels))
        loss_grad = grad(loss)

        for _ in range(steps):
            sample = np.random.choice(np.arange(train_labels.shape[0]), batch_size)
            sample_inputs = train_inputs[sample]
            sample_labels = train_labels[sample]
            self.weights = self.weights - learning_rate * loss_grad(self.weights, sample_inputs, sample_labels)

    def compute_inference(self, input_str):
        last_activation = 0
        results = []

        for j in range(len(input_str)):
            start_idx, end_idx = self._get_index_ranges(j)
            pre_padding, post_padding = self._get_padding(start_idx, end_idx, len(input_str))
            vector = self._string_vectorizer(input_str[max(0, start_idx):end_idx])

            last_activation = SimpleRecurrentModel._raw_inference(
                self.weights,
                np.concatenate((pre_padding, vector, post_padding, [last_activation, 1]))
            )
            results.append(last_activation)

        return results

    def _string_vectorizer(self, string):
        return np_utils.to_categorical(list(map(self.alphabet.get, string)), len(self.alphabet))

    def _get_index_ranges(self, inference_index):
        return inference_index - self.window_shift, inference_index + self.window_size - self.window_shift

    def _get_padding(self, start_idx, end_idx, len_input_str):
        pre_padding = np.zeros(-min(0, start_idx) * len(self.alphabet))
        post_padding = np.zeros(max(0, end_idx - len_input_str) * len(self.alphabet))
        return pre_padding, post_padding

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            dill.dump(self, f)

    def assemble_data(self, inputs, labels):
        assert len(inputs) == len(labels)

        formatted_inputs = []
        formatted_labels = []

        for i in range(len(inputs)):
            last_label = False
            vectorized_string = self._string_vectorizer(inputs[i])

            for j in range(len(inputs[i])):
                start_idx, end_idx = self._get_index_ranges(j)
                pre_padding, post_padding = self._get_padding(start_idx, end_idx, len(inputs[i]))
                vector = self._string_vectorizer(inputs[i][max(0, start_idx):end_idx]).flatten()
                formatted_inputs.append(np.concatenate((pre_padding, vector, post_padding, [last_label, 1])))
                last_label = any([j >= interval[0] and j < interval[1] for interval in labels[i]])
                formatted_labels.append(int(last_label))

        return np.array(formatted_inputs), np.array(formatted_labels)

    @staticmethod
    def _raw_inference(params, inputs):
        output = np.matmul(inputs, params)
        return np.exp(output) / (1 + np.exp(output))
