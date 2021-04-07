class Optimizer(object):
    """
    Base class for a neural network optimizer.
    """

    def __init__(self, lr: float = 0.01):
        """
        Every optimizer must have an initial learning rate.
        """
        self.lr = lr

    def step(self) -> None:
        """
        Every optimizer must implement the "step" function.
        """
        pass


class SGD(Optimizer):
    """
    Stochasitc gradient descent optimizer.
    """

    def __init__(self, lr: float = 0.01) -> None:
        """Pass"""
        super().__init__(lr)

    def step(self):
        """
        For each parameter, adjust in the appropriate direction, with the magnitude of the adjustment
        based on the learning rate.
        """
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad
