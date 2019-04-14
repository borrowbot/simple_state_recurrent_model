import autograd.numpy as np
from autograd import grad
import string
import math
import os
import json


DEFAULT_ALPHABET = string.printable[:95] + '€£'


def load_saved_model(model_path):
    weights = np.load(os.path.join(model_path, 'weights.npy'))
    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    model = SimpleRecurrentModel(**metadata)
    model.weights = weights
    return model


class SimpleRecurrentModel(object):
    def __init__(self, window_size, alphabet=DEFAULT_ALPHABET, window_shift=0):
        self.num_chars = len(alphabet)
        self.window_size = window_size
        self.alphabet = alphabet
        self.window_shift = window_shift
        self.weights = np.random.normal(1, size=(self.num_chars * self.window_size + 2))


    def train(self, train_inputs, train_labels, batch_size, learning_rate, steps):
        def loss(params, sample_inputs, sample_labels):
            output = SimpleRecurrentModel._raw_inference(params, sample_inputs)
            return -np.sum(output * sample_labels + (1 - output) * (1 - sample_labels))
        loss_grad = grad(loss)

        train_inputs, train_labels = self.assemble_data(train_inputs, train_labels)
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
        vector = [
            [0 if char != letter else 1 for char in self.alphabet]
            for letter in string
        ]
        return np.array(vector).flatten()


    def _get_index_ranges(self, inference_index):
        return inference_index - self.window_shift, inference_index + self.window_size - self.window_shift


    def _get_padding(self, start_idx, end_idx, len_input_str):
        pre_padding = np.zeros(-min(0, start_idx) * len(self.alphabet))
        post_padding = np.zeros(max(0, end_idx - len_input_str) * len(self.alphabet))
        return pre_padding, post_padding


    def save(self, file_dir):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        np.save(os.path.join(file_dir, 'weights.npy'), self.weights)

        with open(os.path.join(file_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'window_size': self.window_size,
                'alphabet': self.alphabet,
                'window_shift': self.window_shift
            }, f)


    def assemble_data(self, inputs, labels):
        assert len(inputs) == len(labels)

        formatted_inputs = []
        formatted_labels = []

        for i in range(len(inputs)):
            last_label = False
            for j in range(len(inputs[i])):
                start_idx, end_idx = self._get_index_ranges(j)
                pre_padding, post_padding = self._get_padding(start_idx, end_idx, len(inputs[i]))
                vector = self._string_vectorizer(inputs[i][max(0, start_idx):end_idx])

                formatted_inputs.append(np.concatenate((pre_padding, vector, post_padding, [last_label, 1])))
                last_label = any([j >= interval[0] and j < interval[1] for interval in labels[i]])
                formatted_labels.append(int(last_label))

        return np.array(formatted_inputs), np.array(formatted_labels)


    @staticmethod
    def _raw_inference(params, inputs):
        output = np.matmul(inputs, params)
        return np.exp(output) / (1 + np.exp(output))
