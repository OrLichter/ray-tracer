import torch

from surfaces.sphere import Sphere
from ray import Rays

def test_sphere_ray_intersection():
    sphere = Sphere([0, 0, 0], 1, 0)
    rays = Rays(
        origins=torch.tensor([[0, 0, 2], [0, 0, 2], [0, 0, 2]]),
        directions=torch.tensor([[0, 0, -1], [0, 0, 1], [0, 1, 0]])
    )
    points = sphere.ray_intersect(rays)
    expected_points = torch.tensor([0, 0, 1], dtype=torch.float32)
    assert torch.allclose(points[0], expected_points)    
    assert torch.isnan(points[1:]).all()

if __name__ == '__main__':
    test_sphere_ray_intersection()
    print("All tests passed!")