import torch
from functools import cached_property
from ray import Rays
from typing import Tuple

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = torch.tensor(position, dtype=torch.float32)
        self.look_at = torch.tensor(look_at, dtype=torch.float32)
        self.up_vector = torch.tensor(up_vector, dtype=torch.float32)
        self.screen_distance = torch.tensor(screen_distance, dtype=torch.float32)
        self.screen_width = torch.tensor(screen_width, dtype=torch.float32)

    def world_to_cam(self) -> torch.Tensor:
        """Computes the world-to-camera transformation matrix.

        This function calculates a 4x4 transformation matrix that converts
        world space coordinates to camera space coordinates. It uses the
        camera's position, look-at point, and up vector to construct the
        camera's local coordinate system.
        As convention, -z is front, y is up, and x is right.

        The resulting matrix combines both the rotation to align the world
        axes with the camera's axes and the translation to position the
        world origin at the camera's position.

        Returns:
            torch.Tensor: A 4x4 transformation matrix that converts world
                coordinates to camera coordinates when applied to homogeneous
                coordinates (x, y, z, 1).

        Example:
            camera = Camera([0, 0, 5], [0, 0, 0], [0, 1, 0], 1, 2)
            world_to_cam_matrix = camera.world_to_cam()
            world_point = torch.tensor([1, 2, 3, 1], dtype=torch.float32)
            cam_point = torch.matmul(world_to_cam_matrix, world_point)
        """
        # Calculate the camera's coordinate system
        forward = self.direction
        
        right = torch.cross(forward, self.up_vector)
        right = right / torch.norm(right)
        
        up = torch.cross(right, forward)
        
        # Create the rotation matrix
        rotation = torch.stack([
            torch.cat([right, torch.tensor([0.0])]),
            torch.cat([up, torch.tensor([0.0])]),
            torch.cat([-forward, torch.tensor([0.0])]),
            torch.tensor([0.0, 0.0, 0.0, 1.0])
        ])
        
        # Create the translation matrix
        translation = torch.eye(4)
        translation[:3, 3] = -self.position
        
        return rotation @ translation


    def generate_rays(self, loc: Tuple[int, int], width: int, height: int) -> Rays:
        """Generate a ray for a pixel at coordinates (i, j).

        This function generates a ray that starts at the camera's position
        and passes through the pixel at coordinates (i, j) on the camera's
        screen. The screen is located at the camera's look-at point, and its
        width and height are determined by the camera's screen width and the
        aspect ratio of the image.

        Args:
            i (int): The x-coordinate of the pixel.
            j (int): The y-coordinate of the pixel.

        Returns:
            Rays: A Rays object representing the generated ray.

        Example:
            camera = Camera([0, 0, 5], [0, 0, 0], [0, 1, 0], 1, 2)
            rays = camera.generate_rays(0, 0)
        """
        i, j = loc

        # Calculate the aspect ratio of the image
        aspect_ratio = width / height

        # Calculate the pixel's position on the screen
        x = (i + 0.5) / width
        y = (j + 0.5) / height

        # Transform to camera space
        screen_y = ((2 * x - 1) * aspect_ratio * self.screen_width / 2)
        screen_x = (1 - 2 * y) * self.screen_height(aspect_ratio) / 2

        # Calculate the direction of the ray in camera space
        # direction = torch.tensor([screen_x, screen_y, self.screen_distance], device=self.position.device)
        direction = torch.tensor([screen_x, screen_y, self.screen_distance], device=self.position.device)
        direction = direction / torch.norm(direction)

        # Transform the direction from camera to world space
        world_to_cam = self.world_to_cam()
        direction = torch.matmul(world_to_cam[:3, :3], direction)
        
        return Rays(self.position, direction)
    
    
    @cached_property
    def direction(self) -> torch.Tensor:
        # Calculate the camera's coordinate system
        forward = self.look_at - self.position
        forward = forward / torch.norm(forward)

        return forward

    def screen_height(self, aspect_ratio: float) -> torch.Tensor:
        return self.screen_width / aspect_ratio