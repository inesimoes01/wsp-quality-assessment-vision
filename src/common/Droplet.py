class Droplet:
    def __init__(self, center_x:int, center_y:int, radius:float, id:int,  overlappedIDs):
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.overlappedIDs = overlappedIDs

