class Droplet:
    def __init__(self, center_x, center_y, radius, id=None,  overlappedIDs=None, color=None):
        if id is not None and overlappedIDs is not None and color is not None:
            self.id = id
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.overlappedIDs = overlappedIDs
            self.color = color
        elif color is not None:
            self.id = id
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.overlappedIDs = overlappedIDs
        else:         
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
    
    def print_droplet(self):
        print("Center: ", self.center_x, " ", self.center_y, " Radius: ", self.radius)
    

