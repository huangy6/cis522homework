import torch
from typing import Callable


class MLP(torch.nn.Module):
    """Implementation of a simple multi-layer perceptron"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.activation = activation
        self.initializer = initializer

        # Declare layers
        layers = [torch.nn.Linear(input_size, hidden_size)]
        for hidden in range(hidden_count - 1):
            layers += [torch.nn.Linear(hidden_size, hidden_size)]
        layers += [torch.nn.Linear(hidden_size, num_classes)]
        self.layers = torch.nn.ModuleList(layers)

        # initialize layers
        [self.initializer(l.weight) for l in self.layers]
        print(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.layers[0](x)

        if len(self.layers) > 2:
            for i in range(1, len(self.layers) - 1):
                x = self.layers[i](x)
                x = self.activation(x)

        x = self.layers[-1](x)
        return x
