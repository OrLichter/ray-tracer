import torch

from surfaces.primitive import Primitive
from ray import Rays

class Cube(Primitive):
    def __init__(self, position, scale, material_index):
        self.position = torch.tensor(position)
        self.scale = scale
        self.material_index = material_index

    def transform_(self, matrix: torch.Tensor):
        self.position = self.position @ matrix.T
    
    def ray_intersect(self, rays: Rays) -> torch.Tensor:
        """
        Compute intersection of rays with the cube, using SLAB method
        
        Args:
            rays: Rays object
        
        Returns:
            points: torch.Tensor of shape (n, 3) of intersection points
        """
        # Compute intersection with axis-aligned cube
        t_min = (self.position - 0.5 * self.scale - rays.origins) / rays.directions
        t_max = (self.position + 0.5 * self.scale - rays.origins) / rays.directions

        t_near = torch.max(torch.min(t_min, t_max), dim=1)[0]
        t_far = torch.min(torch.max(t_min, t_max), dim=1)[0]

        # Check if intersection occurs
        valid_intersections = (t_near < t_far) & (t_far > 0)

        # Compute intersection points
        t = torch.where(valid_intersections, t_near, torch.tensor(float('inf')))

        points = rays(t)
        
        valid = torch.logical_and(valid_intersections, t >= 0)
        points[~valid] = float('nan')
        
        return points