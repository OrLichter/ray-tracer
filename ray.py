import torch
from typing import Union
from dataclasses import dataclass

@dataclass
class Ray:
    origin: torch.Tensor
    direction: torch.Tensor

    def __call__(self, t):
        return self.origin + t * self.direction


@dataclass
class Rays:
    origins: torch.Tensor
    directions: torch.Tensor
    
    def __init__(self, origins: torch.Tensor, directions: torch.Tensor):
        if len(origins.shape) == 1:
            origins = origins[None]
        if len(directions.shape) == 1:
            directions = directions[None]
        self.origins = origins
        self.directions = directions
        
    def __getitem__(self, key):
        return Ray(self.origins[key], self.directions[key])
    
    def __call__(self, t: Union[torch.Tensor, float]):
        return self.origins + t[..., None] * self.directions