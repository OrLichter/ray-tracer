from functools import cached_property
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
        position_homogeneous = matrix @ position_homogeneous
        self.position = position_homogeneous[:3]
    
    @cached_property
    def nan_points_array(self):
        return torch.full((1, 3), float('nan'))

    @cached_property
    def normal_points_array(self):
        return torch.full((1, 3), float('nan'))

    @cached_property
    def ones_array(self):
        return torch.ones(1)
    
    @cached_property
    def radius_squared(self):
        return self.radius * self.radius

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

        a = self.ones_array.repeat(oc.shape[0])
        b = 2.0 * torch.sum(oc * rays.directions, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - self.radius_squared
        
        discriminant = b * b - 4 * a * c
        
        points = self.nan_points_array.repeat(rays.origins.shape[0], 1)
        normals = self.normal_points_array.repeat(rays.origins.shape[0], 1)
        
        # Compute the smaller root for valid intersections (- discriminant is necassarily the smaller root)
        t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
        
        valid = torch.logical_and(discriminant >= 0, t >= 0)

        # Assign valid intersection points to the output tensor
        if valid.any():
            valid_points = rays(t)[valid]
            points[valid] = valid_points
            normals[valid] = (valid_points - self.position) / self.radius
    
        return points, t, normals

    def aabb(self) -> torch.Tensor:
        """
        Compute the axis-aligned bounding box (AABB) of the sphere.
        
        Returns:
            torch.Tensor: A tensor of shape (2, 3) representing the AABB of the sphere.
        """

        min_corner = self.position - self.radius
        max_corner = self.position + self.radius

        aabb = torch.stack([min_corner, max_corner])
        
        return aabb