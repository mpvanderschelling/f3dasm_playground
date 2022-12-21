from functools import partial
from typing import List

import numpy as np
import tensorflow as tf


def get_reshaped_array_from_list_of_arrays(flat_array: np.ndarray, list_of_arrays: List[np.ndarray]) ->  List[np.ndarray]:
    total_array = []
    index = 0
    for mimic_array in list_of_arrays:
        number_of_values = np.product(mimic_array.shape)
        current_array = np.array(flat_array[index:index+number_of_values])

        if number_of_values > 1:
            current_array = current_array.reshape(-1,1) # Make 2D array

        total_array.append(current_array)
        index += number_of_values

    return total_array

def get_flat_array_from_list_of_arrays(list_of_arrays: List[np.ndarray]) -> List[np.ndarray]: # technically not a np array input!
    return np.concatenate([np.atleast_2d(array) for array in list_of_arrays])

class MLArchitecture(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        # self.model.add(tf.keras.layers.InputLayer(input_shape=(dimensionality,)))

    def _forward(self, X):
        assert hasattr(self, 'model'), 'model is defined'
        return self.model(X)

    def call(self, X, *args, **kwargs): # Shape: (samples, dim)
        return self._forward(X, *args)

    def loss(self, Y_pred, Y_true): #Y_hat = model output, Y = labels
        return self._loss_function(Y_pred, Y_true)

    def _loss_function(Y_pred, Y_true):
        raise NotImplementedError

    def get_model_weights(self) -> List[np.ndarray]:
        return get_flat_array_from_list_of_arrays(self.model.get_weights())
        # return self.model.get_weights()

    def set_model_weights(self, weights: np.ndarray):
        reshaped_weights = get_reshaped_array_from_list_of_arrays(flat_array=weights.ravel(), list_of_arrays=self.model.get_weights())
        self.model.set_weights(reshaped_weights)

    # RECONSIDER THESE METHODS

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])


# -------------------------------------------------------------

class SimpleModel(MLArchitecture):
    def __init__(self, loss_function): # introduce loss_function parameter because no data to compare to!
        super().__init__()

        # Loss function is a benchmark function
        self._loss_function = loss_function

        # We don't have labels for benchmark function loss
        self.loss = partial(self.loss, Y_true=None)

    def get_weights():
        return NotImplementedError("There are no trainable weights to for this type of models!")

    def set_weights():
        return NotImplementedError("There are no trainable weights to for this type of models!")

# -------------------------------------------------------------

class LinearRegression(MLArchitecture):
    def __init__(self, dimensionality: int): # introduce a dimensionality parameter because trainable weights!
        super().__init__()
        self.model.add(tf.keras.layers.Dense(1, input_shape=(dimensionality,)))

    def _loss_function(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)