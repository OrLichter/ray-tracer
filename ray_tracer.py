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
from surfaces.primitive import Primitive



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
        self.width = width
        self.height = height
        self._transform_scene_to_view()
        
    def _transform_scene_to_view(self):
        world_to_cam_matrix = self.camera.world_to_cam()
        for obj in self.objects:
            obj.transform_(world_to_cam_matrix)

    def render(self) -> np.ndarray:
        """ Render the scene """
        image = torch.zeros((self.width, self.height, 3))
        for i in range(self.width): # TODO: return batch of all rays
            for j in range(self.height):
                # Generate multiple rays per pixel
                rays = self.camera.generate_rays((i, j), self.width, self.height)

                depth = self.scene_settings.max_recursions
                color = self.trace_rays(rays, depth)
                image[j, i] += color
                print("DEBUG::Rendering pixel ({}, {})".format(i, j))

        return (image * 255).clamp(0, 255).cpu().numpy().astype(np.uint8) # W x H x 3

    def trace_rays(self, rays: Rays, depth: int):
        """ Trace the rays and return the color of the pixel """
        closest_intersection_distance = np.inf
        closest_obj = None
        closest_intersection = None

        for obj in self.objects:
            if isinstance(obj, Light):
                continue
            intersection_point, distances, normals = obj.ray_intersect(rays)  # TODO: Should be batched, for now only one ray
            if (~intersection_point.isnan()).sum() > 0: # if there is an intersection
                if distances[0] < closest_intersection_distance:
                    closest_intersection_distance = distances[0]
                    closest_intersection = intersection_point[0]
                    closest_obj = obj
                    closest_normal = normals[0]

        if closest_intersection is None:
            return self.scene_settings.background_color

        # Compute the color
        color = self.compute_color(closest_obj, closest_intersection, rays, closest_normal, depth)

        return color
    
    def compute_color(
        self,
        obj: Primitive,
        intersection: torch.Tensor,
        rays: torch.Tensor,
        normal: torch.Tensor,
        depth
    ):
        """ Compute the color of the pixel """
        if depth <= 0:
            return torch.zeros(3)

        color = torch.zeros(3)
        diffuse_color = obj.material.diffuse_color
        specular_color = obj.material.specular_color
        reflection_color = obj.material.reflection_color
        shininess = obj.material.shininess
        transparency = obj.material.transparency

        # Accumulate contributions from all light sources
        for light in self.objects:
            if not isinstance(light, Light):
                continue

            # Calculate light direction and distance
            light_vec = light.position - intersection
            light_distance = torch.norm(light_vec)
            light_dir = light_vec / light_distance

            # Check for shadows
            shadow_ray = Rays(intersection + normal * 1e-4, light_dir)
            shadow_factor = 1.0
            for shadow_obj in self.objects:
                if shadow_obj == obj or shadow_obj == light:
                    continue
                shadow_intersection, shadow_distances, _ = shadow_obj.ray_intersect(shadow_ray)
                if not shadow_intersection.isnan().all():
                    if shadow_distances[0] < light_distance:
                        shadow_factor = 1.0 - light.shadow_intensity
                    break

            if shadow_factor > 0:
                # Diffuse component
                nl_dot = torch.clamp(torch.dot(normal, light_dir), 0, 1)
                diffuse = diffuse_color * nl_dot * light.color * shadow_factor

                # Specular component
                view_dir = -rays.directions[0]
                reflect_dir = 2 * torch.dot(normal, light_dir) * normal - light_dir
                spec_dot = torch.clamp(torch.dot(view_dir, reflect_dir), 0, 1)
                specular = specular_color * (spec_dot ** shininess) * light.color * light.specular_intensity * shadow_factor

                # Add contribution from this light
                color += diffuse + specular

        # Reflection
        if torch.any(reflection_color > 0) and depth > 0:
            reflect_dir = rays.directions[0] - 2 * torch.dot(rays.directions[0], normal) * normal
            reflect_ray = Rays(intersection + normal * 1e-4, reflect_dir)
            reflect_color = self.trace_rays(reflect_ray, depth - 1)
            color += reflection_color * reflect_color

        # Refraction (transparency)
        if transparency > 0 and depth > 0:
            # Simplified refraction (assuming same refractive index for all materials)
            refract_dir = self.refract(rays.directions,[0], normal, 1.0, 1.5)  # Assuming air to glass
            if refract_dir is not None:
                refract_ray = Rays(intersection - normal * 1e-4, refract_dir)
                refract_color = self.trace_rays(refract_ray, depth - 1)
                color = color * (1 - transparency) + refract_color * transparency

        return torch.clamp(color, 0, 1)

    def refract(incident: torch.Tensor, normal: torch.Tensor, n1: float, n2: float) -> torch.Tensor:
        """Calculate the refraction direction."""
        ratio = n1 / n2
        cos_i = -torch.dot(normal, incident)
        sin_t_sq = ratio * ratio * (1.0 - cos_i * cos_i)
        if sin_t_sq > 1.0:
            return None  # Total internal reflection
        cos_t = torch.sqrt(1.0 - sin_t_sq)
        return ratio * incident + (ratio * cos_i - cos_t) * normal


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
    parser.add_argument('--scene_file', default="scenes/pool.txt", type=str, help='Path to the scene file')
    parser.add_argument('--output_image', default="test.png", type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=128, help='Image width')
    parser.add_argument('--height', type=int, default=128, help='Image height')
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
