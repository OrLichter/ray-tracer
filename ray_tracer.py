import argparse
from PIL import Image
import numpy as np
import torch
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Rays


class RayTracer:
    """ pytorch based ray tracer """
    def __init__(self, camera, scene_settings, objects, width, height):
        self.camera = camera
        self.scene_settings = scene_settings
        self.objects = objects
        self.width = width
        self.height = height

    def render(self):
        image = torch.zeros((self.width, self.height, 3))
        for i in range(self.width):
            for j in range(self.height):
                # Generate multiple rays per pixel
                rays = self.camera.generate_rays(i, j)

                depth = 0 # TODO: implement depth
                color = self.trace_rays(rays, depth)
                image[i, j] += color

        return image

    def trace_rays(self, rays: Rays, depth: int):
        # Find the closest intersection
        closest_intersection = None
        for obj in self.objects:
            intersection = obj.ray_intersect(rays)
            if intersection is not None:
                if closest_intersection is None or intersection.t < closest_intersection.t:
                    closest_intersection = intersection

        if closest_intersection is None:
            return torch.zeros(3)

        # Compute the color
        color = self.compute_color(closest_intersection, rays, depth)

        return color

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', default="/Users/orlichter/Documents/School/Masters/Computer Graphics/hw2/scenes/pool.txt", type=str, help='Path to the scene file')
    parser.add_argument('output_image', default="test.png", type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    tracer = RayTracer(camera, scene_settings, objects, args.width, args.height)
    image_array = tracer.render()
    image_array = image_array.cpu().numpy()

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
