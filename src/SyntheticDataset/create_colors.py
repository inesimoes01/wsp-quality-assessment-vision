class Colors:
    def __init__(self, outside_color, light_color, dark_color, background_colors):
        self.generate_color(outside_color, light_color, dark_color, background_colors)


    def generate_color(self, outside_color, light_color, dark_color, background_colors):
        self.outside_color = [self.hex_to_rgb(color) for color in outside_color]
        self.light_color = [self.hex_to_rgb(color) for color in light_color]
        self.dark_rgb = [self.hex_to_rgb(color) for color in dark_color]

        # BACKGROUND COLOR IN RGB
        self.background_colors = [self.hex_to_rgb(color) for color in background_colors]
        

    def hex_to_rgb(self, hex_code):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return rgb


