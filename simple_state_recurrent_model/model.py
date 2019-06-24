import autograd.numpy as np
from autograd import grad
import string
import math
import os
import json
import dill
from keras.utils import np_utils


def load_saved_model(model_path):
    """ Loads a SimpleRecurrentModel from a serialized file.
    """
    with open(model_path, 'rb') as f:
        return dill.load(f)


class SimpleRecurrentModel(object):
    """ An implementation of a simple, lightweight ARMA(1, window_size) type logistic regression language model for
        detecting references to simple semantic entities in text. A written description and example application of this
        model can be found on our blog at (todo).

    Args:
        alphabet <str>: The list of characters recognized by the model. Characters in the data which are not in the
            alphabet are encoded as a zero vector.
        preprocess: A function mapping a string to a string which is applied to inputs (including training data) before
            processing by the model.
        window_size <int>: See (todo) for a thorough description of this parameter.
        window_shift <int>: See (todo) for a thorough description of this parameter.
    """
    def __init__(self, window_size, alphabet, window_shift=0, preprocess=lambda x: x):
        self.num_chars = len(alphabet) + 1
        self.window_size = window_size
        self.alphabet = {alphabet[i]: i + 1 for i in range(len(alphabet))}
        self.alphabet_map = lambda x: self.alphabet.get(x, 0)
        self.window_shift = window_shift
        self.preprocess = preprocess
        self.weights = np.zeros(shape=(self.num_chars * self.window_size + 2), dtype='float32')

    def train(self, train_inputs, train_labels, batch_size, learning_rate, steps):
        """ Trains the model weights by using stochastic gradient descent to maximize the naive likelihood.

        Args
            train_inputs <list>: A list of input strings in their raw form i.e. before preprocessing and conversion into
                input vectors.
            train_labels <list>: A list identifying each occurance of a semantic entity for each string in
                `train_inputs`. See (todo) for a description of the expected format of this parameter.
            batch_size <int>: The size of the batch used in stochastic gradient descent.
            learning_rate <float>: The step size multiplier used in stochastic gradient descent.
            steps <int>: The number of training steps to take during the function's execution.
        """
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
        """ Returns the result of the model's inference on an input string. Returns a list floats with the same length
            as the original input string. Values in this array are between 0 and 1.
        """
        last_activation = 0
        results = []
        vectorized_string = self._string_vectorizer(filtered_input_str)
        for j in range(len(filtered_input_str)):
            start_idx, end_idx = self._get_index_ranges(j)
            pre_padding, post_padding = self._get_padding(start_idx, end_idx, len(filtered_input_str))
            vector = vectorized_string[max(0, start_idx):end_idx].flatten()

            last_activation = SimpleRecurrentModel._raw_inference(
                self.weights,
                np.concatenate((pre_padding, vector, post_padding, [last_activation, 1]))
            )
            results.append(last_activation)

        return results

    def _string_vectorizer(self, string):
        index_vectors = list(map(self.alphabet.get, string))
        return np_utils.to_categorical(index_vectors, self.num_chars)

    def _get_index_ranges(self, inference_index):
        return inference_index - self.window_shift, inference_index + self.window_size - self.window_shift

    def _get_padding(self, start_idx, end_idx, len_input_str):
        pre_padding = np.zeros(-min(0, start_idx) * self.num_chars)
        post_padding = np.zeros(max(0, end_idx - len_input_str) * self.num_chars)
        return pre_padding, post_padding

    def save(self, file_path):
        """ Serializes the model represented by an instance of the SimpleRecurrentModel class into a file which can be
            recovered using the `load_saved_model` function.
        """
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
