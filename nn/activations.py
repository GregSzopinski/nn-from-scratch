import numpy as np
from nn.operations import Operation


class Linear(Operation):
    """
    "Identity" activation function
    """

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self) -> np.ndarray:
        """Pass through"""
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Pass through"""
        return output_grad


class Sigmoid(Operation):
    """
    Sigmoid activation function.
    """

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self) -> np.ndarray:
        """
        Compute output.
        """
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Compute input gradient.
        """
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class ReLU(Operation):
    """
    Rectified Linear Unit activation function
    """

    def __init__(self) -> None:
        super.__init__()

    def _output(self) -> np.ndarray:
        return np.max(self.input_, 0)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """Compute input gradient"""
        relu_backward = self.output >= 0
        input_grad = relu_backward * output_grad
        return input_grad


class Tanh(Operation):
    """
    Hyperbolic tangent activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self, inference: bool) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - self.output * self.output)
