from surfaces.primitive import Primitive
import torch

class InfinitePlane(Primitive):
    def __init__(self, normal, offset, material_index, material=None):
        self.normal = torch.as_tensor(normal)
        self.offset = torch.as_tensor(offset)
        self.material_index = torch.as_tensor(material_index)
        self.material = material

    def transform_(self, matrix):
        self.normal = self.normal @ matrix.T
        self.offset = self.offset @ matrix.T
    
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
        points = rays(t)
        
        return points

    @property
    def color(self):
        return self.material.diffuse_color