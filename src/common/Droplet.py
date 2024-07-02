class Droplet:
    def __init__(self, center_x:int, center_y:int, area:float, id:int,  overlappedIDs, radius:float = None):
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.area = area
        self.overlappedIDs = overlappedIDs

