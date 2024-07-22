import torch
from surfaces.sphere import Sphere

class Light(Sphere):
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        super().__init__(position, radius, 0)
        self.position = torch.as_tensor(position)
        self._color = torch.as_tensor(color)
        self.specular_intensity = torch.as_tensor(specular_intensity)
        self.shadow_intensity = torch.as_tensor(shadow_intensity)
        self.radius = torch.as_tensor(radius)

    @property
    def color(self):
        return self._color