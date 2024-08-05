class Scene:
    def __init__(
        self,
        scene_settings,
        camera,
        objects
    ):
        self.scene_settings = scene_settings
        self.camera = camera
        self.objects = objects

        world_to_cam_matrix = self.camera.world_to_cam
        for obj in self.objects:
            obj.transform_(world_to_cam_matrix)