import torch

class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = torch.as_tensor(diffuse_color)
        self.specular_color = torch.as_tensor(specular_color)
        self.reflection_color = torch.as_tensor(reflection_color)
        self.shininess = torch.as_tensor(shininess)
        self.transparency = torch.as_tensor(transparency)
