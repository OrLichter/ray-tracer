from surfaces.primitive import Primitive


class InfinitePlane(Primitive):
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index
