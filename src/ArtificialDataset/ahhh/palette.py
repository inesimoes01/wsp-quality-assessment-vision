import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb, rgb_to_hsv
import colorsys
import numpy as np
import cv2

brown_colors = [ '#190e00', '#1f150c', '#0e0f13', '#131514', '#1b0f19', '#14141c', '#1b141c', '#131522', '#0c0d22', '#181123', '#141325', '#1c1527', '#181729', '#1b1a2a']
dark_blue_colors=['#09082a', '#0f0c2b', '#181130', '#0a0a30', '#030430', '#160e33', '#000233', '#191935', '#0e0d35', '#0e0d35', '#140c35', '#181736', '#060838', '#060838', '#060a3a', '#060a3a', '#11143d', '#0d0b3d', '#0e1040', '#030444', '#060845', '#0d0b4a', '#0b094a', '#070654']
light_blue_color = ['#181872', '#18107f', '#221d91', '#272595', '#2c2897', '#352ea0', '#2d29a2', '#2524ac', '#2e2db5', '#2c2bb7']

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return rgb

brown_rgb = [hex_to_rgb(color) for color in brown_colors]
light_blue_rgb = [hex_to_rgb(color) for color in light_blue_color]
dark_blue_rgb = [hex_to_rgb(color) for color in dark_blue_colors]

# # Plotting the colors in order of lightness
# plt.figure(figsize=(14, 6))

# # Plot brown colors
# for i, color in enumerate(brown_colors):
#     plt.scatter(i, 0, color=color, s=500)

# # Plot blue colors
# for i, color in enumerate(light_blue_color):
#     plt.scatter(i, 1, color=color, s=500)

# for i, color in enumerate(dark_blue_colors):
#     plt.scatter(i, 2, color=color, s=500)

# plt.yticks([0, 1, 2], ['Brown', 'Light Blue', 'Dark Blue'])
# plt.title('Ordered Colors by Lightness')
# plt.gca().axes.get_xaxis().set_visible(False)
# plt.show()

def interpolate_color(color1, color2, t):
    """ Interpolate between color1 and color2 based on parameter t (0 to 1). """
    r = int(color1[0] * (1 - t) + color2[0] * t)
    g = int(color1[1] * (1 - t) + color2[1] * t)
    b = int(color1[2] * (1 - t) + color2[2] * t)
    return (r, g, b)

def draw_smooth_color_circle(img, center, radius, inner_dark_colors, inner_light_colors, outer_colors):

    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                # distance from center normalized to [0, 1]
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2) / radius
                
                if distance < 0.6:  # inner part (0 to 60% of the radius)
                    t = distance / 0.6  # scale t to [0, 1]
                    color_index = int(t * (len(inner_light_colors) - 1))
                    t = (t * (len(inner_light_colors) - 1)) - color_index
                    color1 = inner_light_colors[color_index % len(inner_light_colors)]
                    color2 = inner_light_colors[(color_index + 1) % len(inner_light_colors)]
                    interpolated_color = interpolate_color(color1, color2, t)
                
                elif distance < 0.7:  # middle part (60% to 90% of the radius)
                    t = (distance - 0.6) / 0.3  
                    inner_index = int((distance / 0.6) * (len(inner_light_colors) - 1))
                    middle_index = int((t) * (len(inner_dark_colors) - 1))
                    
                    # Calculate interpolation between inner and middle colors
                    t_inner = (distance / 0.6 * (len(inner_light_colors) - 1)) - inner_index
                    t_middle = (t * (len(inner_dark_colors) - 1)) - middle_index
                    
                    inner_color1 = inner_light_colors[inner_index % len(inner_light_colors)]
                    inner_color2 = inner_light_colors[(inner_index + 1) % len(inner_light_colors)]
                    middle_color1 = inner_dark_colors[middle_index % len(inner_dark_colors)]
                    middle_color2 = inner_dark_colors[(middle_index + 1) % len(inner_dark_colors)]
                    
                    inner_interpolated = interpolate_color(inner_color1, inner_color2, t_inner)
                    middle_interpolated = interpolate_color(middle_color1, middle_color2, t_middle)
                    
                    interpolated_color = interpolate_color(inner_interpolated, middle_interpolated, t)

                elif distance < 0.9:  # Middle part (60% to 90% of the radius)
                    t = (distance - 0.7) / 0.3  # Scale t to [0, 1]
                    color_index = int(t * (len(inner_dark_colors) - 1))
                    t = (t * (len(inner_dark_colors) - 1)) - color_index
                    color1 = inner_dark_colors[color_index % len(inner_dark_colors)]
                    color2 = inner_dark_colors[(color_index + 1) % len(inner_dark_colors)]
                    interpolated_color = interpolate_color(color1, color2, t)
                
                else:  # outer part (90% to 100% of the radius)
                    t = (distance - 0.8) / 0.1  # Scale t to [0, 1]
                    middle_index = int(((distance - 0.6) / 0.3) * (len(inner_dark_colors) - 1))
                    outer_index = int(t * (len(outer_colors) - 1))
                    
                    # Calculate interpolation between middle and outer colors
                    t_middle = ((distance - 0.6) / 0.3 * (len(inner_dark_colors) - 1)) - middle_index
                    t_outer = (t * (len(outer_colors) - 1)) - outer_index
                    
                    middle_color1 = inner_dark_colors[middle_index % len(inner_dark_colors)]
                    middle_color2 = inner_dark_colors[(middle_index + 1) % len(inner_dark_colors)]
                    outer_color1 = outer_colors[outer_index % len(outer_colors)]
                    outer_color2 = outer_colors[(outer_index + 1) % len(outer_colors)]
                    
                    middle_interpolated = interpolate_color(middle_color1, middle_color2, t_middle)
                    outer_interpolated = interpolate_color(outer_color1, outer_color2, t_outer)
                    
                    interpolated_color = interpolate_color(middle_interpolated, outer_interpolated, t)

                if distance > 0.95: # make sure to blend with background
                    t = (distance - 0.95) / 0.1  
                    color_index = int(t * (len(outer_colors) - 1))
                    color1 = outer_colors[color_index % len(outer_colors)]
                    color2 = (255, 255, 0) 
                    interpolated_color = interpolate_color(color1, color2, t)
               
                # set pixel color in the image
                img[y, x] = interpolated_color

# Create a blank image
img = np.zeros((512, 512, 3), dtype=np.uint8)
img = cv2.rectangle(img, (0, 0), (512, 512), (255, 255, 0), cv2.FILLED)
# Define circle parameters
center = (256, 256)
radius = 15

# Draw the varied color circle
draw_smooth_color_circle(img, center, radius, dark_blue_rgb, light_blue_rgb, brown_rgb)
plt.imshow(img)
plt.show()
