class Colors:
    def __init__(self):
        self.generate_color()

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

    def generate_color(self):

        brown_colors = [ '#190e00', '#1f150c', '#0e0f13', '#131514', '#1b0f19', '#14141c', '#1b141c', '#131522', '#0c0d22', '#181123', '#141325', '#1c1527', '#181729', '#1b1a2a']
        dark_blue_colors=['#09082a', '#0f0c2b', '#181130', '#0a0a30', '#030430', '#160e33', '#000233', '#191935', '#0e0d35', '#0e0d35', '#140c35', '#181736', '#060838', '#060838', '#060a3a', '#060a3a', '#11143d', '#0d0b3d', '#0e1040', '#030444', '#060845', '#0d0b4a', '#0b094a', '#070654']
        light_blue_color = ['#2c2bb7', '#2e2db5', '#2524ac', '#2d29a2', '#352ea0', '#2c2897', '#272595', '#221d91', '#18107f', '#181872']
        
        self.brown_rgb = [self.hex_to_rgb(color) for color in brown_colors]
        self.light_blue_rgb = [self.hex_to_rgb(color) for color in light_blue_color]
        self.dark_blue_rgb = [self.hex_to_rgb(color) for color in dark_blue_colors]
        
        # DROPLET COLOR IN BGR
        self.droplet_color_small = []
        self.droplet_color_big = []
        # self.droplet_color_big.extend(self.interpolate_color((29, 33, 52), (61, 42, 64), 20))
        # self.droplet_color_small.extend(self.interpolate_color((89, 8, 37), (172, 4, 46), 20))

        # BACKGROUND COLOR IN RGB
        self.background_colors =[]
        background_color_1 = (255, 244, 137)
        background_color_2 = (159, 127, 19)
        self.background_colors.extend(self.interpolate_color(background_color_1, background_color_2, 30))

    def hex_to_rgb(self, hex_code):
        hex_code = hex_code.lstrip('#')
        rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return rgb

