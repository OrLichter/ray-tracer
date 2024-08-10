from functools import lru_cache
from PIL import Image
import argparse
import numpy as np
import torch

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Rays
from surfaces.primitive import Primitive
from time import time


RESOLUTION = 128
GRID_SIZE = 10
NUM_WORKERS = cpu_count() - 1
DEBUG = True

def is_valid_object(obj):
    is_sphere = isinstance(obj, Sphere)
    is_plane = isinstance(obj, InfinitePlane)
    is_cube = isinstance(obj, Cube)
    is_light = isinstance(obj, Light)

    return any([is_sphere, is_plane, is_cube, is_light])

class RayTracer:
    """ pytorch based ray tracer """
    def __init__(self, camera: Camera, scene_settings: SceneSettings, objects: int, width: int, height: int):
        self.camera = camera
        self.scene_settings = scene_settings
        self.objects = [obj for obj in objects if is_valid_object(obj)]
        self.non_light_objects = [obj for obj in self.objects if not isinstance(obj, Light)]
        self.lights = [obj for obj in self.objects if isinstance(obj, Light)]
        self.width = width
        self.height = height
        self._transform_scene_to_view()
        self.generate_aabb()
        self.build_scene_grid(grid_size=GRID_SIZE)
        
    def _transform_scene_to_view(self):
        world_to_cam_matrix = self.camera.world_to_cam
        for obj in self.objects:
            obj.transform_(world_to_cam_matrix)

    def generate_aabb(self):
        """ Generate scene bounding box """
        self.aabb = torch.zeros(2, 3)
        objs_aabb = [obj.aabb() for obj in self.non_light_objects]
        objs_aabb = [obj for obj in objs_aabb if obj is not None]

        self.aabb[0] = torch.min(torch.stack([obj[0] for obj in objs_aabb]))
        self.aabb[1] = torch.max(torch.stack([obj[1] for obj in objs_aabb]))

    def build_scene_grid(self, grid_size: int):
        """ Build a grid for the scene, based on self.aabb for faster intersection checks """
        self.grid_size = grid_size
        # Create a grid where each cell contains a list to hold object indices
        self.grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid = [[[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Compute the dimensions of the AABB
        aabb_size = self.aabb[1] - self.aabb[0]

        # For each object, find the grid cells it intersects and add it to the grid
        for obj_index, obj in enumerate(self.non_light_objects):
            if isinstance(obj, InfinitePlane):
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        for k in range(self.grid_size):
                            # find the intersection of the plane with the grid cell
                            cell_min = self.aabb[0] + torch.tensor([i, j, k]) * aabb_size / self.grid_size
                            cell_max = cell_min + aabb_size / self.grid_size
                            intersection, _, _ = obj.ray_intersect(Rays(cell_min, cell_max - cell_min))
                            if not intersection.isnan().all():
                                self.grid[i][j][k].append(obj_index)
                continue


            aabb = obj.aabb()

            # Calculate min and max cell indices for the object's AABB
            min_cell = torch.floor((aabb[0] - self.aabb[0]) / aabb_size * self.grid_size).int()
            max_cell = torch.floor((aabb[1] - self.aabb[0]) / aabb_size * self.grid_size).int()

            # Ensure cell indices are clamped within the grid bounds
            min_cell = torch.clamp(min_cell, 0, self.grid_size - 1)
            max_cell = torch.clamp(max_cell, 0, self.grid_size - 1)

            # Iterate over the grid cells the object intersects
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    for k in range(min_cell[2], max_cell[2] + 1):
                        # Append the object index to the list for the current grid cell
                        self.grid[i][j][k].append(obj_index)

    @lru_cache
    def get_objects_in_cell(self, cell):
        """ Return the list of object indices in the specified grid cell """
        x, y, z = cell
        return self.grid[x][y][z]  # Returns a list of object indices in the specified cell

    def get_cell_from_ray(self, ray: Rays):
        """ Return the grid cell that the ray intersects """
        # Compute the intersection point of the ray with the scene AABB
        t_min = (self.aabb[0] - ray.origins) / ray.directions
        t_max = (self.aabb[1] - ray.origins) / ray.directions

        # Compute the entry and exit points of the ray with the scene AABB
        t_enter = torch.max(torch.min(t_min, t_max), dim=-1)[0]
        t_exit = torch.min(torch.max(t_min, t_max), dim=-1)[0]

        # Compute the intersection points of the ray with the scene AABB
        enter_point = ray(t_enter)
        exit_point = ray(t_exit)

        # Compute the cell indices for the entry and exit points
        cell_size = (self.aabb[1] - self.aabb[0]) / self.grid_size
        
        enter_cell = torch.floor((enter_point - self.aabb[0]) / cell_size).int()
        exit_cell = torch.floor((exit_point - self.aabb[0]) / cell_size).int()

        # Clamp the cell indices to the grid bounds
        enter_cell = torch.clamp(enter_cell, 0, self.grid_size - 1)
        exit_cell = torch.clamp(exit_cell, 0, self.grid_size - 1)

        return enter_cell.squeeze(), exit_cell.squeeze()

    def get_relevant_cells(self, entry_cell, exit_cell):
        # Ensure entry_cell and exit_cell are the correct shape
        entry_cell = entry_cell.reshape(-1, 3)
        exit_cell = exit_cell.reshape(-1, 3)
        
        # Calculate the direction of traversal
        step = torch.sign(exit_cell - entry_cell)
        
        # Initialize the list of relevant cells
        relevant_cells = []
        
        # Iterate over all rays
        for i in range(len(entry_cell)):
            current_cell = entry_cell[i].clone()
            relevant_cells.append(current_cell.tolist())
            
            # Check if entry and exit cells are the same
            if torch.all(current_cell == exit_cell[i]):
                continue
            
            # Traverse the grid until we reach the exit cell
            while not torch.all(current_cell == exit_cell[i]):
                # Determine which dimension to step in
                # We step in the dimension that has the smallest t value
                t = torch.where(
                    step[i] != 0,
                    (current_cell - entry_cell[i] + 0.5 * step[i]) / (exit_cell[i] - entry_cell[i]),
                    float('inf')
                )
                dim = torch.argmin(t)
                
                # Take a step in that dimension
                current_cell[dim] += step[i, dim]
                
                # Add the new cell to the list
                relevant_cells.append(current_cell.tolist())

        # Remove duplicates and return
        relevant_cells = list(dict.fromkeys(list(tuple(cell) for cell in relevant_cells)))
        return relevant_cells

    def process_pixel(self, i, j):
        rays = self.camera.generate_rays((i, j), self.width, self.height)
        color = self.trace_rays(rays, self.scene_settings.max_recursions)
        return color

    def process_row(self, args):
        i, row_pixels = args
        result = np.zeros((len(row_pixels), 3))  # Assuming RGB image
        for idx, j in enumerate(row_pixels):
            result[idx] = self.process_pixel(i, j)
        return i, result
    

    def render(self) -> np.ndarray:
        """ Render the scene """
        image = np.zeros((self.width, self.height, 3))
        pixel_indices = [(i, j) for i in range(self.width) for j in range(self.height)]

        if DEBUG:
            min_w = self.width // 3
            max_w = 2 * self.width // 3
            min_h = self.height // 3
            max_h = 2 * self.height // 3
            pixel_indices = [(i, j) for i in range(min_w, max_w) for j in range(min_h, max_h)]

        rows = {}
        for i, j in pixel_indices:
            if i not in rows:
                rows[i] = []
            rows[i].append(j)
        row_tasks = [(i, rows[i]) for i in sorted(rows.keys())]

        if DEBUG:
            for row_task in tqdm(row_tasks):
                i, result = self.process_row(row_task)
                image[i][:len(result)] = result
        else:
            with Pool(NUM_WORKERS) as pool:
                for i, result in tqdm(pool.imap(self.process_row, row_tasks), total=len(row_tasks), desc="Rendering rows"):
                    image[i, :len(result)] = result

        image = image.transpose(1, 0, 2)  # H x W x 3

        return (image * 255).clip(0, 255).astype(np.uint8) # W x H x 3

    def trace_rays(self, rays: Rays, depth: int):
        """ Trace the rays and return the color of the pixel """

        if depth == 0:
            return self.scene_settings.background_color

        closest_intersection_distance = np.inf
        closest_obj = None
        closest_intersection = None

        entry_cell, exit_cell = self.get_cell_from_ray(rays)  # A function to compute the cell index from the ray
        relevant_cells = self.get_relevant_cells(entry_cell, exit_cell)

        object_to_run_intersect = []
        for cell in relevant_cells:
            objects_in_cell = self.get_objects_in_cell(cell)
            object_to_run_intersect.extend(objects_in_cell)

        # find first occurence for each element in the list object_to_run_intersect
        object_to_run_intersect = list(dict.fromkeys(object_to_run_intersect))[::-1]

        for obj_idx in object_to_run_intersect:
            obj = self.non_light_objects[obj_idx]
            intersection_point, distances, normals = obj.ray_intersect(rays)  # TODO: Should be batched, for now only one ray
            if (~intersection_point.isnan()).sum() > 0: # if there is an intersection
                if distances[0] < closest_intersection_distance:
                    closest_intersection_distance = distances[0]
                    closest_intersection = intersection_point[0]
                    closest_obj = obj
                    closest_normal = normals[0]
                    # break

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
        depth: int
    ):
        """ Compute the color of the pixel """

        transparency = obj.material.transparency
        color = self.scene_settings.background_color * transparency

        for light in self.lights:
            average_shadow_factor = self.compute_soft_shadows_batched(obj, normal, intersection, light, self.scene_settings.root_number_shadow_rays)
            color += self.add_diffuse_and_specular(obj, rays, normal, intersection, light, average_shadow_factor)

        color += self.add_reflection(obj, intersection, rays, normal, depth)
        color = self.add_transparency(obj, intersection, rays, normal, depth, color)

        return torch.clamp(color, 0, 1)

    def compute_hard_shadows(self, obj, normal, intersection, light):
        light_vec = light.position - intersection
        light_distance = torch.norm(light_vec)
        light_dir = light_vec / light_distance

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
        return shadow_factor
    
    def compute_soft_shadows_batched(self, obj, normal, intersection, light, grid_size):      
        # Compute light vector and distance
        light_vec = light.position - intersection
        light_distance = torch.norm(light_vec)
        light_dir = light_vec / light_distance
        
        # Create a plane perpendicular to the light direction
        plane_normal = light_dir
        plane_point = light.position
        
        # Create two orthogonal vectors on the plane
        u = torch.cross(plane_normal, torch.tensor([1.0, 0.0, 0.0]))
        if torch.norm(u) < 1e-6:
            u = torch.cross(plane_normal, torch.tensor([0.0, 1.0, 0.0]))
        u = u / torch.norm(u)
        v = torch.cross(plane_normal, u)
        
        # Calculate the grid size
        cell_size = 2 * light.radius / grid_size
        
        # Generate grid coordinates
        grid_coords = torch.stack(torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), 
                                                 indexing="ij"), dim=-1).float()
        grid_coords = grid_coords.to(intersection.device)
        
        # Reshape grid_coords to (grid_size*grid_size, 2)
        grid_coords = grid_coords.view(-1, 2)
        
        # Generate random offsets for each cell
        random_offsets = torch.rand(grid_size*grid_size, 2, device=intersection.device)
        
        # Calculate points on the plane for all cells
        cell_points = (grid_coords + random_offsets) * cell_size - light.radius
        points_on_plane = plane_point.unsqueeze(0) + cell_points[:,0].unsqueeze(-1) * u.unsqueeze(0) + cell_points[:,1].unsqueeze(-1) * v.unsqueeze(0)
        
        # Create shadow rays from the points on the plane to the intersection
        shadow_ray_dirs = intersection.unsqueeze(0) - points_on_plane
        shadow_ray_dirs = shadow_ray_dirs / torch.norm(shadow_ray_dirs, dim=-1, keepdim=True)
        shadow_rays = Rays(points_on_plane + normal.unsqueeze(0) * 1e-4, shadow_ray_dirs)
        
        # Check for intersections
        hit = torch.zeros(grid_size*grid_size, dtype=torch.bool, device=intersection.device)
        for shadow_obj in self.objects:
            if shadow_obj == obj or shadow_obj == light:
                continue
            shadow_intersections, shadow_distances, _ = shadow_obj.ray_intersect(shadow_rays)
            valid_intersections = ~torch.isnan(shadow_intersections).all(dim=-1)
            closer_intersections = shadow_distances < light_distance
            hit |= valid_intersections & closer_intersections
        
        rays_hit = (~hit).float().sum()
        
        shadow_factor = (1 - light.shadow_intensity) + light.shadow_intensity * (rays_hit / (grid_size * grid_size))
        return shadow_factor

    def add_diffuse_and_specular(self, obj, rays, normal, intersection, light, average_shadow_factor):
        light_vec = light.position - intersection
        light_distance = torch.norm(light_vec)
        light_dir = light_vec / light_distance
        if average_shadow_factor > 0:
            # Diffuse component
            nl_dot = torch.clamp(torch.dot(normal, light_dir), 0, 1)
            diffuse = obj.material.diffuse_color * nl_dot * light.color * average_shadow_factor

            # Specular component
            view_dir = -rays.directions[0]
            reflect_dir = 2 * torch.dot(normal, light_dir) * normal - light_dir
            spec_dot = torch.clamp(torch.dot(view_dir, reflect_dir), 0, 1)
            specular = obj.material.specular_color * (spec_dot ** obj.material.shininess) * light.color * light.specular_intensity * average_shadow_factor

            # Add contribution from this light
            return (diffuse + specular) * (1 - obj.material.transparency)
        return 0

    def add_reflection(self, obj, intersection, rays, normal, depth):
        if torch.any(obj.material.reflection_color > 0) and depth > 0:
            reflect_dir = rays.directions[0] - 2 * torch.dot(rays.directions[0], normal) * normal
            reflect_ray = Rays(intersection + normal * 1e-4, reflect_dir)
            reflect_color = self.trace_rays(reflect_ray, depth - 1)
            return obj.material.reflection_color * reflect_color
        return 0

    def add_transparency(self, obj, intersection, rays, normal, depth, color):
        if obj.material.transparency > 0 and depth > 0:
            # Simplified refraction (assuming same refractive index for all materials)
            refract_dir = self.refract(rays.directions[0], normal, 1.0, 1.5)  # Assuming air to glass
            if refract_dir is not None:
                refract_ray = Rays(intersection - normal * 1e-4, refract_dir)
                refract_color = self.trace_rays(refract_ray, depth - 1)
                return color * (1 - obj.material.transparency) + refract_color * obj.material.transparency
        return color

    def refract(self, incident: torch.Tensor, normal: torch.Tensor, n1: float, n2: float) -> torch.Tensor:
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


def save_image(image_array, output_path):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('--scene_file', default="scenes/poolcube.txt", type=str, help='Path to the scene file')
    parser.add_argument('--output_image', default="output/test.png", type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=RESOLUTION, help='Image width')
    parser.add_argument('--height', type=int, default=RESOLUTION, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    tracer = RayTracer(camera, scene_settings, objects, args.width, args.height)
    start = time()
    image_array = tracer.render()
    print("Time taken to render: {:.2f} seconds".format(time() - start))

    # Dummy result
    # image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
