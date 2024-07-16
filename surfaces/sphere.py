import torch
from ray import Rays
from surfaces.primitive import Primitive

class Sphere(Primitive):
    def __init__(self, position, radius, material_index):
        self.position = torch.tensor(position)
        self.radius = radius
        self.material_index = material_index

    def transform_(self, matrix: torch.Tensor):
        self.position = self.position @ matrix.T
    
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

        # Initialize the output tensor with NaNs
        points = torch.full((rays.origins.shape[0], 3), float('nan'), device=rays.origins.device)

        # Compute the smaller root for valid intersections
        t = (-b - torch.sqrt(discriminant)) / (2.0 * a)

        valid = torch.logical_and(discriminant >= 0, t >= 0)
        
        # Compute intersection points for valid intersections
        valid_points = rays[valid](t[valid])
        
        # Assign valid intersection points to the output tensor
        points[valid] = valid_points

        return points