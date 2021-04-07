import numpy as np


def assert_same_shape(array: np.ndarray, array_grad: np.ndarray):
    assert array.shape == array_grad.shape, \
        """
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        """.format(tuple(array_grad.shape), tuple(array.shape))
    return None


class Operation(object):
    """
    Base class for an "operation" in a neural network.
    """

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray):
        """
        Stores input in the self._input instance variable
        Calls the self._output() function.
        """
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Calls the self._input_grad() function.
        Checks that the appropriate shapes match.
        """
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> np.ndarray:
        """
        The _output method must be defined for each Operation
        """
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        The _input_grad method must be defined for each Operation
        """
        raise NotImplementedError()


class ParamOperation(Operation):
    """
    An Operation with parameters.
    """

    def __init__(self, param: np.ndarray) -> np.ndarray:
        """
        The ParamOperation method
        """
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        """

        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Every subclass of ParamOperation must implement _param_grad.
        """
        raise NotImplementedError()
