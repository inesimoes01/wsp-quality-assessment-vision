class Droplet:
    def __init__(self, isElipse:bool, center_x:int, center_y:int, radius:float, id:int,  overlappedIDs, color=None):
        if color is not None:
            self.id = id
            self.isElipse = isElipse
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.overlappedIDs = overlappedIDs
            self.color = color
        else:
            self.id = id
            self.isElipse = isElipse
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.overlappedIDs = overlappedIDs

