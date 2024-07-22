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
        # oc = self.position - rays.origins

        num_rays = rays.origins.shape[0] if len(rays.origins.shape) > 1 else 1
        a = torch.ones(num_rays, device=rays.origins.device).squeeze()
        b = 2.0 * oc @ rays.directions
        c = torch.sum(oc * oc, dim=-1) - self.radius ** 2

        discriminant = b * b - 4 * a * c

        # Initialize the output tensor with NaNs
        points = torch.full((num_rays, 3), float('nan'), device=rays.origins.device).squeeze()

        # Compute the smaller root for valid intersections
        sqrt_discriminant = torch.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)
        t1_valid = torch.logical_and(t1 >= 0, discriminant >= 0)
        t2_valid = torch.logical_and(t2 >= 0, discriminant >= 0)

        t = torch.where(t1_valid, t1, torch.where(t2_valid, t2, torch.tensor(float('inf'), device=rays.origins.device)))

        valid = torch.logical_and(discriminant >= 0, t < float('inf'))

        # Compute intersection points for valid intersections
        valid_points = rays[valid](t[valid])

        # Assign valid intersection points to the output tensor
        if valid.any():
            points[valid] = valid_points
        else:
            #TODO: (Or) return all nans
            points = EMPTY_TENSOR.to(rays.origins.device)

        return points
    
    @property
    def color(self):
        return self.material.diffuse_color