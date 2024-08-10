from surfaces.primitive import Primitive
import torch

class InfinitePlane(Primitive):
    def __init__(self, normal, offset, material_index, material=None):
        self.normal = torch.as_tensor(normal)
        self.normal = self.normal / torch.norm(self.normal)
        self.offset = torch.as_tensor(offset)
        self.material_index = torch.as_tensor(material_index)
        self.material = material

    def transform_(self, matrix):
        """
        Apply a 4x4 transformation matrix to the planes's normal and offset.
        
        Args:
        matrix (torch.Tensor): A 4x4 transformation matrix.
        """
        assert matrix.shape == (4, 4), "Transformation matrix must be 4x4"

        rotation_matrix = matrix[:3, :3]
        translation_vector = matrix[:3, 3]

        # Transform the normal vector
        transformed_normal = rotation_matrix @ self.normal

        # Adjust the offset using the transformed normal and the translation part
        transformed_offset = self.offset + transformed_normal @ translation_vector

        # Update the plane's normal and offset
        self.normal = transformed_normal
        self.offset = transformed_offset
    
    def ray_intersect(self, rays):
        """
        Compute intersection of rays with the infinite plane
        
        Args:
            rays: Rays object
        
        Returns:
            points: torch.Tensor of shape (n, 3) of intersection points
        """
        # Compute the intersection points

        direction_dot_normal = rays.directions @ self.normal
        point_on_plane = self.normal * self.offset
        vector_to_plane = point_on_plane - rays.origins
        t = (vector_to_plane @ self.normal) / direction_dot_normal

        points = rays(t)
        points[t < 0] = float('nan')
        
        return points, t, self.normal.expand_as(points)

    def aabb(self):
        return None
    
    @property
    def name(self):
        return "InfinitePlane"
