import tensorflow as tf

from mlclasses import Module


class Classifier(Module):  #@save
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = tf.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = tf.cast(tf.argmax(Y_hat, axis=1), Y.dtype)
        compare = tf.cast(preds == tf.reshape(Y, -1), tf.float32)
        return tf.reduce_mean(compare) if averaged else compare



class SoftmaxRegression(Classifier):  #@save
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.num_outputs = num_outputs
        self.lr = lr

        self.net = tf.keras.models.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = tf.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = tf.reshape(Y, (-1,))
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)