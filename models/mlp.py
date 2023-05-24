import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, activation, norm, bias=True,
    ):
        """Basic MLP block

        Args:
            in_features (int): size of input vector
            out_features (int): size of output vector
            norm : norm layer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.bias = bias

        layers = [nn.Linear(in_features, out_features, bias=bias)]
        if norm is not None:
            layers.append(norm(num_features=out_features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.activation is not None:
            x = self.activation(x)
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
        bias=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth
        self.residual = residual
        self.activation = activation
        self.bias = bias

        layers = [LinearBlock(input_size, width, activation, norm, bias)]
        for i in range(depth - 1):
            layers.append(LinearBlock(width, width, activation, norm, bias))
        layers.append(nn.Linear(width, output_size, bias))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, save_activations=False):
        if save_activations:
            self.activations = []

        for i, layer in enumerate(self.layers):
            if (layer.in_features == layer.out_features) and self.residual:
                y = layer(x)
                if save_activations:
                    self.activations.append(y)
                x = x + y
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
