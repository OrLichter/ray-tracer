from functools import cached_property
import torch

from surfaces.primitive import Primitive
from ray import Rays

class Cube(Primitive):
    def __init__(self, position, scale, material_index):
        self.position = torch.tensor(position, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.material_index = material_index
        self.rotation = torch.eye(3)  # Identity matrix for initial rotation
        self.update_corners()

    def update_corners(self):
        half_size = 0.5 * self.scale
        local_corners = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ], dtype=torch.float32) * half_size
        rotated_corners = torch.matmul(local_corners, self.rotation.T)
        self.corners = rotated_corners + self.position

    def rotate(self, rotation_matrix: torch.Tensor):
        """
        Apply a 3x3 rotation matrix to the cube.
        Args:
        rotation_matrix (torch.Tensor): A 3x3 rotation matrix.
        """
        assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3"
        self.rotation = torch.matmul(rotation_matrix, self.rotation)
        self.update_corners()

    def transform_(self, matrix: torch.Tensor):
        """
        Apply a 4x4 transformation matrix to the cube.
        Args:
        matrix (torch.Tensor): A 4x4 transformation matrix.
        """
        assert matrix.shape == (4, 4), "Transformation matrix must be 4x4"
        
        # Transform the position
        homogeneous_position = torch.cat([self.position, torch.ones(1)])
        transformed_position = torch.matmul(matrix, homogeneous_position)
        self.position = transformed_position[:3] / transformed_position[3]
        
        # Extract rotation from the transformation matrix
        rotation = matrix[:3, :3]
        scale_factors = torch.norm(rotation, dim=0)
        normalized_rotation = rotation / scale_factors
        
        # Update rotation
        self.rotation = torch.matmul(normalized_rotation, self.rotation)
        
        # Update scale
        self.scale *= scale_factors.mean()
        
        # Update corners
        self.update_corners()

    def ray_intersect(self, rays: Rays):
        """
        Compute the intersection of rays with the cube.
        
        Args:
        rays (Rays): An object containing ray origins and directions.
        
        Returns:
        tuple: (hit_points, distances, normals)
            hit_points: tensor of shape (N, 3) with the intersection points (xyz), or NaN if no hit
            distances: tensor of shape (N,) with the distances to intersections, or NaN if no hit
            normals: tensor of shape (N, 3) with the normals at intersection points, or NaN if no hit
        """
        # Transform rays to local cube space
        local_origins = rays.origins - self.position
        local_origins = torch.matmul(local_origins, self.rotation)
        local_directions = torch.matmul(rays.directions, self.rotation)
        
        # Compute intersections with axis-aligned planes in local space
        t_min = (-0.5 * self.scale - local_origins) / local_directions
        t_max = (0.5 * self.scale - local_origins) / local_directions
        
        t_near = torch.max(torch.min(t_min, t_max), dim=1).values
        t_far = torch.min(torch.max(t_min, t_max), dim=1).values
        
        # Check if there's a valid intersection
        mask = (t_near < t_far) & (t_far > 0)
        distances = torch.where(mask, t_near, torch.full_like(t_near, float('nan')))
        
        # Compute hit points
        local_hit_points = local_origins + distances.unsqueeze(1) * local_directions
        hit_points = torch.matmul(local_hit_points, self.rotation.T) + self.position
        
        # Compute normals
        eps = 1e-6
        local_normals = torch.zeros_like(local_hit_points)
        local_normals[torch.abs(local_hit_points[:, 0] - 0.5 * self.scale) < eps, 0] = 1
        local_normals[torch.abs(local_hit_points[:, 0] + 0.5 * self.scale) < eps, 0] = -1
        local_normals[torch.abs(local_hit_points[:, 1] - 0.5 * self.scale) < eps, 1] = 1
        local_normals[torch.abs(local_hit_points[:, 1] + 0.5 * self.scale) < eps, 1] = -1
        local_normals[torch.abs(local_hit_points[:, 2] - 0.5 * self.scale) < eps, 2] = 1
        local_normals[torch.abs(local_hit_points[:, 2] + 0.5 * self.scale) < eps, 2] = -1
        normals = torch.matmul(local_normals, self.rotation.T)
        
        # Apply mask to hit_points and normals
        hit_points = torch.where(mask.unsqueeze(1), hit_points, torch.full_like(hit_points, float('nan')))
        normals = torch.where(mask.unsqueeze(1), normals, torch.full_like(normals, float('nan')))
        
        return hit_points, distances, normals

    @property
    def min_corner(self) -> torch.Tensor:
        """
        Compute the minimum corner of the cube.
        
        Returns:
        torch.Tensor: A tensor of shape (3,) representing the minimum corner.
        """
        return torch.min(self.corners, dim=0).values
    
    @property
    def max_corner(self) -> torch.Tensor:
        """
        Compute the maximum corner of the cube.
        
        Returns:
        torch.Tensor: A tensor of shape (3,) representing the maximum corner.
        """
        return torch.max(self.corners, dim=0).values

    def aabb(self) -> torch.Tensor:
        """
        Compute the Axis-Aligned Bounding Box (AABB) of the cube.
        
        Returns:
        torch.Tensor: A tensor of shape (2, 3) representing the min and max corners of the AABB.
        """

        return torch.stack([self.min_corner, self.max_corner])