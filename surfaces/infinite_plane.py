from surfaces.primitive import Primitive
import torch

class InfinitePlane(Primitive):
    def __init__(self, normal, offset, material_index, material=None):
        self.normal = torch.as_tensor(normal)
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
        #TODO (Or): Implement this a bit differently
        t = -(torch.sum(rays.origins * self.normal, dim=-1) + self.offset) / torch.sum(rays.directions * self.normal, dim=-1)
        if t < 0:
            points = torch.full((rays.origins.shape[0], 3), float('nan'), device=rays.origins.device)
        else:
            points = rays(t)
        
        return points, t, self.normal.expand_as(points)

    @property
    def color(self):
        return self.material.diffuse_color