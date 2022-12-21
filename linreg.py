import tensorflow as tf

from mlclasses import Module


class LinearRegression(Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)

    def forward(self, X):
        """The linear regression model."""
        return self.net(X)

    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)

    def get_w_b(self):
        return (self.get_weights()[0], self.get_weights()[1])