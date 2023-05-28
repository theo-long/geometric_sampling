import torch
import numpy as np


class ModelDistribution:
    def __init__(self, model, pass_as_xy, final_activation=torch.sigmoid):
        self.model = model
        self.final_activation = final_activation
        self.pass_as_xy = pass_as_xy

    def pdf(self, x):
        with torch.no_grad():
            if self.pass_as_xy:
                return self.final_activation(
                    self.model(torch.tensor([np.cos(x), np.sin(x)], dtype=torch.float32))
                ).numpy()[0]
            return self.final_activation(self.model(torch.tensor([x]))).numpy()[0]
