import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, activation, norm
    ):
        """Basic MLP block

        Args:
            input_size (int): size of input vector
            output_size (int): size of output vector
            norm : norm layer
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            norm(num_features=output_size),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        width,
        depth,
        activation=nn.functional.relu,
        norm=nn.LayerNorm,
        residual: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth
        self.residual = residual
        self.activation = activation

        layers = [LinearBlock(input_size, width, norm)]
        for i in range(depth - 1):
            layers.append(LinearBlock(width, width, norm))
        layers.append(nn.Linear(width, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations = []

        for i, layer in enumerate(self.layers):
            if (layer.input_size == layer.output_size) and self.residual:
                y = layer(x)
                if save_activations:
                    self.activations.append(y)
                x = x + self.activation(y)
            else:
                x = layer(x)
        return x


class MultiHeadMLP(MLP):
    def __init__(
        self,
        input_size,
        output_size,
        width,
        depth,
        head_depth,
        num_heads,
        activation=nn.functional.relu,
        norm=nn.LayerNorm,
        residual: bool = False,
    ):
        super().__init__(input_size, width, width, depth)

        self.backbone = MLP(input_size, width, width, depth, activation, norm, residual)
        self.heads = []
        self.num_heads = num_heads
        self.head_depth = head_depth
        self.residual = residual
        for i in range(num_heads):
            self.heads.append(
                MLP(width, output_size, width, head_depth, activation, norm, residual)
            )
        self.heads = nn.ModuleList(self.heads)

    def forward(self, x):
        x = self.backbone(x)

        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        return torch.cat(outputs, axis=0)
