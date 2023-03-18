import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, activation, norm, residual: bool
    ):
        """Basic MLP block

        Args:
            input_size (int): size of input vector
            output_size (int): size of output vector
            activation : activation layer
            norm : norm layer
            residual (bool): residual connection
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            norm(num_features=output_size),
            activation(),
        )

    def forward(self, x):
        out = self.layers(x)
        if self.residual:
            out += x
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        width,
        depth,
        activation=nn.ReLU,
        norm=nn.LayerNorm,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth

        layers = [MLPBlock(input_size, width, activation, norm, residual)]
        for i in range(depth - 1):
            layers.append(MLPBlock(width, width, activation, norm, residual))
        layers.append(nn.Linear(width, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % (len(self.layers) - 1) == 0:
                x = layer(x)
            else:
                x = x + layer(x)
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
        activation=nn.ReLU,
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
