import torch
from abc import ABC, abstractmethod
from typing import overload
from ray import Rays

#TODO: (Or) remove this
EMPTY_TENSOR = torch.empty((0, 3))

class Primitive(ABC):
    @overload
    def __init__(self, position, scale, material_index):
        pass

    @overload
    def __init__(self, normal, offset, material_index):
        pass
    
    @overload
    def __init__(self, position, radius, material_index):
        pass

    @abstractmethod
    def transform_(self, matrix: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def aabb(self):
        """ Compute the axis-aligned bounding box of the primitive """
        raise NotImplementedError

    @abstractmethod
    def ray_intersect(rays: Rays) -> torch.Tensor:
        """
        Compute intersection of rays with the sphere
        
        Args:
            rays: Rays object
        
        Returns:
            points: torch.Tensor of shape (n, 3) of intersection points
        """

        raise NotImplementedError
