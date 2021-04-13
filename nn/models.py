import numpy as np
from copy import deepcopy
from typing import List, Tuple
from layers import Layer
from loss import Loss
from optimizers import Optimizer


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


class NeuralNetwork(object):
    """
    The class for a neural network.
    """

    def __init__(self, layers: List[Layer], loss: Loss, seed: int = 1) -> None:
        """
        Neural networks need layers, and a loss.
        """
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Passes data forward through a series of layers.
        """
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Passes data backward through a series of layers.
        """

        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """
        Passes data forward through the layers.
        Computes the loss.
        Passes data backward through the layers.
        """

        predictions = self.forward(x_batch)

        loss = self.loss.forward(predictions, y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        """
        Gets the parameters for the network.
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """
        Gets the gradient of the loss with respect to the parameters for the network.
        """
        for layer in self.layers:
            yield from layer.param_grads


class Trainer(object):
    """
    Trains a neural network
    """

    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        """
        Requires a neural network and an optimizer in order for training to occur.
        Assign the neural network as an instance variable to the optimizer.
        """
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, "net", self.net)

    def generate_batches(
        self, X: np.ndarray, y: np.ndarray, size: int = 32
    ) -> Tuple[np.ndarray]:
        """
        Generates batches for training
        """
        assert (
            X.shape[0] == y.shape[0]
        ), """
            features and target must have the same number of rows, instead
            features has {0} and target has {1}
            """.format(
            X.shape[0], y.shape[0]
        )

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]

            yield X_batch, y_batch

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        restart: bool = True,
    ) -> None:
        """
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        """

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9

        for e in range(epochs):

            if (e + 1) % eval_every == 0:
                # for early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e + 1) % eval_every == 0:

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e + 1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(
                        f"""Loss increased after epoch {e + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {e + 1 - eval_every}"""
                    )
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, "net", self.net)
                    break
