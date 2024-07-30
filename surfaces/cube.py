import torch

from surfaces.primitive import Primitive
from ray import Rays

class Cube(Primitive):
    def __init__(self, position, scale, material_index):
        self.position = torch.tensor(position)
        self.scale = scale
        self.material_index = material_index

    def transform_(self, matrix: torch.Tensor):
        """
        Apply a 4x4 transformation matrix to the planes's normal and offset.
        
        Args:
        matrix (torch.Tensor): A 4x4 transformation matrix.
        """
        assert matrix.shape == (4, 4), "Transformation matrix must be 4x4"

        position_homogeneous = torch.cat([self.position, torch.tensor([1.0])], dim=0)
        
        position_homogeneous = position_homogeneous @ matrix.T
        self.position = position_homogeneous[:3]
    
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
        normals = torch.zeros_like(points)
        # TODO: Compute normals

        
        return points, t, normals