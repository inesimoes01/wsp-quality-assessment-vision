class Colors:
    def __init__(self, outside_color, light_color, dark_color, background_color1, background_color2):
        self.generate_color(outside_color, light_color, dark_color, background_color1, background_color2)

    def interpolate_color(self, color1, color2, steps):
        # extract individual BGR components
        b1, g1, r1 = color1
        b2, g2, r2 = color2
        
        # calculate step size for each component
        b_step = (b2 - b1) / steps
        g_step = (g2 - g1) / steps
        r_step = (r2 - r1) / steps
        
        # generate interpolated colors
        interpolated_colors = []
        for i in range(steps):
            b = round(b1 + i * b_step)
            g = round(g1 + i * g_step)
            r = round(r1 + i * r_step)
            interpolated_colors.append((b, g, r))
        
        return interpolated_colors

    def generate_color(self, outside_color, light_color, dark_color, background_color1, background_color2):
        self.outside_color = [self.hex_to_rgb(color) for color in outside_color]
        self.light_color = [self.hex_to_rgb(color) for color in light_color]
        self.dark_rgb = [self.hex_to_rgb(color) for color in dark_color]

        # BACKGROUND COLOR IN RGB
        self.background_colors =[]
        self.background_colors.extend(self.interpolate_color(background_color1, background_color2, 30))

    def hex_to_rgb(self, hex_code):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return rgb


