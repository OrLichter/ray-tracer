import torch
from camera import Camera


def test_camera_transform_matrix():
    camera = Camera([0, 0, 5],
                    [0, 0, 0],
                    [0, 1, 0], 1, 2)
    world_to_cam_matrix = camera.world_to_cam()
    world_point = torch.tensor([[0, 0, 0, 1],
                                [1, 2, 3, 1]], dtype=torch.float32)
    cam_point =  world_point @ world_to_cam_matrix.T
    expected_point = torch.tensor([[0, 0, -5, 1], 
                                   [1, 2, -2, 1]], dtype=torch.float32)
    assert torch.allclose(cam_point, expected_point, atol=1e-6)
    

def test_camera_transform_matrix2():
    camera = Camera([0, 10, -2],
                    [0, -100, -4],
                    [0, 1, 0], 1, 2)
    world_to_cam_matrix = camera.world_to_cam()
    world_point = torch.tensor([[0, 0, 0, 1],
                                [2, 0, 0, 1]], dtype=torch.float32)
    cam_point =  world_point @ world_to_cam_matrix.T
    expected_point = torch.tensor([[0, -10, -2, 1], 
                                   [2, -10, -2, 1]], dtype=torch.float32)
    assert torch.allclose(cam_point, expected_point, atol=1e-6)
    
    
if __name__ == '__main__':
    test_camera_transform_matrix2()
    print("All tests passed!")