class Droplet:
    def __init__(self, center_x:int, center_y:int, radius:int, id:int,  overlappedIDs, color=None):
        if color is not None:
            self.id = id
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.overlappedIDs = overlappedIDs
            self.color = color
        else:
            self.id = id
            self.center_x = center_x
            self.center_y = center_y
            self.radius = radius
            self.overlappedIDs = overlappedIDs
        # else:         
        #     self.center_x = center_x
        #     self.center_y = center_y
        #     self.radius = radius
    
    def print_droplet(self):
        print("Center: ", self.center_x, " ", self.center_y, " Radius: ", self.radius, " ", self.id, " ", self.overlappedIDs)
    

