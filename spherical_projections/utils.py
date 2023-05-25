import torch
import numpy as np

class ModelDistribution:
    
    def __init__(self, model, final_activation=torch.sigmoid):
        self.model = model
        self.final_activation = final_activation
        
    def pdf(self, x):
        with torch.no_grad():
            return self.final_activation(self.model(torch.tensor([x / (2 * np.pi)]))).numpy()[0]