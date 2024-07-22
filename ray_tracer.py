from PIL import Image
import argparse
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



def is_valid_object(obj):
    is_sphere = isinstance(obj, Sphere)
    is_plane = isinstance(obj, InfinitePlane)
    is_cube = isinstance(obj, Cube)
    is_light = isinstance(obj, Light)

    return any([is_sphere, is_plane, is_cube, is_light])

class RayTracer:
    """ pytorch based ray tracer """
    def __init__(self, camera, scene_settings, objects, width, height):
        self.camera = camera
        self.scene_settings = scene_settings
        self.objects = [obj for obj in objects if is_valid_object(obj)]
        self.objects = self.objects[:2] #TODO: remove hardcoded object - only for debug
        self.width = width
        self.height = height

    def render(self) -> np.ndarray:
        """ Render the scene """
        image = torch.zeros((self.width, self.height, 3))
        for i in range(self.width):
            for j in range(self.height):
                # Generate multiple rays per pixel
                rays = self.camera.generate_rays((i, j), self.width, self.height)

                depth = 0 # TODO: implement depth
                color = self.trace_rays(rays, depth)
                image[i, j] += color
                print("DEBUG::Rendering pixel ({}, {})".format(i, j))

        return (image * 255).clamp(0, 255).cpu().numpy().astype(np.uint8) # W x H x 3

    def trace_rays(self, rays: Rays, depth: int):
        """ Trace the rays and return the color of the pixel """
        closest_intersection_distance = np.inf
        closest_obj = None
        closest_intersection = None

        for obj in self.objects:
            intersection = obj.ray_intersect(rays)
            if len(intersection) != 0: # if there is an intersection
                distance_of_intersection = torch.norm(rays.origins - intersection, dim=-1)
                if distance_of_intersection < closest_intersection_distance:
                    closest_intersection_distance = distance_of_intersection
                    closest_intersection = intersection
                    closest_obj = obj

        if closest_intersection is None:
            return torch.zeros(3)

        # Compute the color
        color = self.compute_color(closest_obj, closest_intersection, rays, depth)

        return color
    
    def compute_color(self, obj, intersection, rays, depth):
        """ Compute the color of the pixel """
        if isinstance(obj, InfinitePlane):
            # return blue
            return obj.color
        
        if isinstance(obj, Sphere):
            # return red
            return obj.color
        

        color = torch.tensor([0, 0, 0])
        # TODO: Implement the color computation
        return color

def parse_scene_file(file_path):
    objects = []
    materials = []
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
                materials.append(material)
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
    
    for obj in objects:
        if hasattr(obj, "material_index"):
            obj.material = materials[obj.material_index-1]
    
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', default="scenes/pool.txt", type=str, help='Path to the scene file')
    parser.add_argument('output_image', default="test.png", type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=8, help='Image width')
    parser.add_argument('--height', type=int, default=8, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    tracer = RayTracer(camera, scene_settings, objects, args.width, args.height)
    image_array = tracer.render()

    # Dummy result
    # image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == '__main__':
    main()
