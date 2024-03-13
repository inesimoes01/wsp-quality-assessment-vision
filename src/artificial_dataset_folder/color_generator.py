def interpolate_color(color1, color2, steps):
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

def generate_color():
    # DROPLET COLOR
    droplet_color = []
    droplet_color.extend(interpolate_color((29, 33, 52), (61, 42, 64), 20))
    droplet_color.extend(interpolate_color((89, 8, 37), (172, 4, 46), 20))

    # BACKGROUND COLOR
    background_color = (97, 225, 243)

    return droplet_color, background_color

