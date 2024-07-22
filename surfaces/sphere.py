import torch
from ray import Rays
from surfaces.primitive import Primitive, EMPTY_TENSOR

class Sphere(Primitive):
    def __init__(self, position, radius, material_index, material=None):
        self.position = torch.tensor(position)
        self.radius = radius
        self.material_index = material_index
        self.material = material

    def transform_(self, matrix: torch.Tensor):
        """
        Apply a 4x4 transformation matrix to the sphere's position.
        
        Args:
        matrix (torch.Tensor): A 4x4 transformation matrix.
        """
        assert matrix.shape == (4, 4), "Transformation matrix must be 4x4"

        position_homogeneous = torch.cat([self.position, torch.tensor([1.0])], dim=0)
        position_homogeneous = position_homogeneous @ matrix.T
        self.position = position_homogeneous[:3]
    
    def ray_intersect(self, rays: Rays) -> torch.Tensor:
        """
        Compute intersection of rays with the sphere
        
        Args:
            rays: Rays object
        
        Returns:
            points: torch.Tensor of shape (n, 3) of intersection points
        """
        # Vector from ray origin to sphere center
        oc = rays.origins - self.position

        a = torch.ones(oc.shape[0], device=rays.origins.device)
        b = 2.0 * torch.sum(oc * rays.directions, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        
        points = torch.full((rays.origins.shape[0], 3), float('nan'), device=rays.origins.device)
        
        # Compute the smaller root for valid intersections (- discriminant is necassarily the smaller root)
        t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
        
        valid = torch.logical_and(discriminant >= 0, t >= 0)


        # Assign valid intersection points to the output tensor
        if valid.any():
            valid_points = rays[valid](t[valid])
            points[valid] = valid_points

        return points
    
    @property
    def color(self):
        return self.material.diffuse_color