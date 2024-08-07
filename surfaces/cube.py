from functools import cached_property
import torch

from surfaces.primitive import Primitive

class Cube(Primitive):
    def __init__(self, position, scale, material_index):
        self.position = torch.tensor(position, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.material_index = material_index
        self.scale = scale

        self.faces_numbers = [
            [0, 1, 3, 2],  # Front face
            [4, 5, 7, 6],  # Back face
            [0, 2, 6, 4],  # Left face
            [1, 3, 7, 5],  # Right face
            [0, 1, 5, 4],  # Bottom face
            [2, 3, 7, 6],  # Top face
        ]

        self.generate_corners()

    def generate_corners(self):
        half_size = self.scale / 2
        corners = torch.tensor([
            [ half_size,  half_size,  half_size],  # (1, 1, 1) * half_size
            [ half_size,  half_size, -half_size],  # (1, 1, -1) * half_size
            [ half_size, -half_size,  half_size],  # (1, -1, 1) * half_size
            [ half_size, -half_size, -half_size],  # (1, -1, -1) * half_size
            [-half_size,  half_size,  half_size],  # (-1, 1, 1) * half_size
            [-half_size,  half_size, -half_size],  # (-1, 1, -1) * half_size
            [-half_size, -half_size,  half_size],  # (-1, -1, 1) * half_size
            [-half_size, -half_size, -half_size],  # (-1, -1, -1) * half_size
        ])
        
        corners += self.position
        self.corners = corners

    def transform_(self, matrix: torch.Tensor):
        assert matrix.shape == (4, 4), "Transformation matrix must be 4x4"
        self.transform_matrix = matrix
        point_homogeneous = torch.cat([self.position, torch.tensor([1.0])], dim=0)
        point_homogeneous = matrix @ point_homogeneous
        self.position = point_homogeneous[:3]

        corners_homogeneous = torch.cat([self.corners, torch.ones((8, 1))], axis=1)
        corners_homogeneous = matrix @ corners_homogeneous.T
        self.corners = corners_homogeneous[:3].T
        self.faces = torch.stack([self.corners[face] for face in self.faces_numbers]).mean(1)

    @property
    def min_corner(self):
        corners = self.corners
        min_corner = torch.min(corners, dim=0)[0]
        return self.position - self.scale / 2
        # return min_corner

    @property
    def max_corner(self):
        return self.position + self.scale / 2
        # corners = self.corners
        # max_corner = torch.max(corners, dim=0)[0]
        # return max_corner

    @cached_property
    def normals(self):
        normals = []

        for face in self.faces_numbers:
            p1, p2, p3, p4 = self.corners[face]
            vec1 = p2 - p1
            vec2 = p4 - p1

            normal = torch.cross(vec1, vec2)
            normal = normal / torch.linalg.norm(normal)
            normals.append(normal)

        return torch.stack(normals)

    @cached_property
    def nan_points_array(self):
        return torch.full((1, 3), float('nan'))

    @cached_property
    def normal_points_array(self):
        return torch.full((1, 3), float('nan'))

    def ray_intersect(self, rays):
        """ Compute intersection of rays with the cube """
        distance_to_min = self.min_corner - rays.origins
        distance_to_max = self.max_corner - rays.origins
        t_min = distance_to_min / (rays.directions+1e-8)
        t_max = distance_to_max / (rays.directions+1e-8)

        t1 = torch.minimum(t_min, t_max)
        t2 = torch.maximum(t_min, t_max)

        t_near = torch.max(t1, dim=1)[0]
        t_far = torch.min(t2, dim=1)[0]

        valid = torch.logical_and(t_near < t_far, t_near > 0)
        points = self.nan_points_array.repeat(rays.origins.shape[0], 1)
        normals = self.normal_points_array.repeat(rays.origins.shape[0], 1)

        if valid.any():
            valid_points = rays(t_near)[valid]
            points[valid] = valid_points

            valid_points_expanded = valid_points[:, None, :]
            faces_expanded = self.faces[None, :, :]

            distances = torch.abs(valid_points_expanded - faces_expanded)
            distances_sum = torch.sum(distances, dim=2)
            
            hit_face = torch.argmin(distances_sum, dim=1)
            hit_normals = self.normals[hit_face] 
            ray_directions = rays.directions[valid] 
            dot_product = torch.sum(hit_normals * ray_directions, dim=1)
            hit_normals[dot_product > 0] *= -1  # Flip the normal if it's pointing in the same direction as the ray

            normals[valid] = hit_normals

        return points, t_near, normals

    def aabb(self) -> torch.Tensor:
        return torch.vstack((self.min_corner, self.max_corner))
