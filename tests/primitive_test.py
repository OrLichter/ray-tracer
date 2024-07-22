import torch

from surfaces.sphere import Sphere
from surfaces.cube import Cube
from ray import Rays

def test_sphere_ray_intersection():
    sphere = Sphere([0, 0, 0], 1, 0)
    rays = Rays(
        origins=torch.tensor([[0, 0, 2], [0, 0, 2], [0, 0, 2]], dtype=torch.float32),
        directions=torch.tensor([[0, 0, -1], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
    )
    points = sphere.ray_intersect(rays)
    expected_points = torch.tensor([0, 0, 1], dtype=torch.float32)
    assert torch.allclose(points[0], expected_points)    
    assert torch.isnan(points[1:]).all()


def test_cube_ray_intersection():
    # Set
    cube = Cube([0, 0, 0], 1, 0)
    cube2 = Cube([0, 0, 5], 2, 0)
    rays = Rays(
        origins=torch.tensor([[0, 0, 2], [0, 1, 1], [0, 0, 2], [0, 0, 2]]),
        directions=torch.tensor([[0, 0, -1], [0, -1, -1],  [0, 0, 1], [0, 1, 0]])
    )
    expected_points1 = torch.tensor([[0, 0, 0.5], [0, 0.5, 0.5]], dtype=torch.float32)
    expected_points2 = torch.tensor([0, 0, 4], dtype=torch.float32)

    # Act
    points = cube.ray_intersect(rays)
    points2 = cube2.ray_intersect(rays)
    
    # Assert
    assert torch.allclose(points[:2], expected_points1)    
    assert torch.isnan(points[2:]).all()

    assert torch.allclose(points2[2], expected_points2)
    assert torch.isnan(points2[:2]).all()
    assert torch.isnan(points2[3]).all()

if __name__ == '__main__':
    test_sphere_ray_intersection()
    print("All tests passed!")