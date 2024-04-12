# from matplotlib import pyplot as plt
# import numpy as np 
# from matplotlib.patches import Rectangle
# from matplotlib.colors import LinearSegmentedColormap

# def hex_to_RGB(hex_str):
#     """ #FFFFFF -> [255,255,255]"""
#     #Pass 16 to the integer function for change of base
#     return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

# def get_color_gradient(c1, c2, n):
#     """
#     Given two hex colors, returns a color gradient
#     with n colors.
#     """
#     assert n > 1
#     c1_rgb = np.array(c1) / 255.0
#     c2_rgb = np.array(c2) / 255.0
#     mix_pcts = [x / (n - 1) for x in range(n)]
#     rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
#     return [(int(round(val[0] * 255)), int(round(val[1] * 255)), int(round(val[2] * 255))) for val in rgb_colors]

# color1 = (97, 225, 243)
# color2 = (97, 225, 30)
# num_points = 5
# color = get_color_gradient(color1, color2, 5)
# print(color)
# image = np.full((76*30, 26*30, 3), color, dtype=np.uint8)
# plt.imshow(image)

# plt.show()

# # colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
# # cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

# # #image = np.full((76*30, 26*30, 3), color, dtype=np.uint8)

# # fig, ax = plt.subplots()
# # rectangle = Rectangle((0, 0), 1, 1, facecolor=cmap(0.5))
# # plt.imshow(rectangle)

# # plt.show()

# # plt.imshow(image)
# # plt.show()

# # import cv2
# # import numpy as np
# # from matplotlib import pyplot as plt 

# # def draw_gradient_alpha_rectangle(frame, BGR_Channel, rectangle_position, rotate):
# #     (xMin, yMin), (xMax, yMax) = rectangle_position
# #     color = np.array(BGR_Channel, np.uint8)[np.newaxis, :]
# #     mask1 = np.rot90(np.repeat(np.tile(np.linspace(1, 0, (rectangle_position[1][1]-rectangle_position[0][1])), ((rectangle_position[1][0]-rectangle_position[0][0]), 1))[:, :, np.newaxis], 3, axis=2), rotate) 
# #     frame[yMin:yMax, xMin:xMax, :] = mask1 * frame[yMin:yMax, xMin:xMax, :] + (1-mask1) * color

# #     return frame

# # frame = np.zeros((300, 300, 3), np.uint8)
# # frame[:,:,:] = 255
# # frame = draw_gradient_alpha_rectangle(frame, (42, 175, 121), ((0, 0), (300, 300)), 3)
# # plt.imshow(frame)
# # plt.show()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


# Draw polygon with linear gradient from point 1 to point 2 and ranging
# from color 1 to color 2 on given image
def linear_gradient(i, poly, p1, p2, c1, c2):

    # Draw initial polygon, alpha channel only, on an empty canvas of image size
    ii = Image.new('RGBA', i.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ii)
    draw.polygon(poly, fill=(0, 0, 0, 255), outline=None)

    # Calculate angle between point 1 and 2
    p1 = np.array(p1)
    p2 = np.array(p2)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) / np.pi * 180

    # Rotate and crop shape
    temp = ii.rotate(angle, expand=True)
    temp = temp.crop(temp.getbbox())
    wt, ht = temp.size

    # Create gradient from color 1 to 2 of appropriate size
    gradient = np.linspace(c1, c2, wt, True).astype(np.uint8)
    gradient = np.tile(gradient, [2 * h, 1, 1])
    gradient = Image.fromarray(gradient)

    # Paste gradient on blank canvas of sufficient size
    temp = Image.new('RGBA', (max(i.size[0], gradient.size[0]),
                              max(i.size[1], gradient.size[1])), (0, 0, 0, 0))
    temp.paste(gradient)
    gradient = temp

    # Rotate and translate gradient appropriately
    x = np.sin(angle * np.pi / 180) * ht
    y = np.cos(angle * np.pi / 180) * ht
    gradient = gradient.rotate(-angle, center=(0, 0),
                               translate=(p1[0] + x, p1[1] - y))

    # Paste gradient on temporary image
    ii.paste(gradient.crop((0, 0, ii.size[0], ii.size[1])), mask=ii)

    # Paste temporary image on actual image
    i.paste(ii, mask=ii)

    return i


# Draw polygon with radial gradient from point to the polygon border
# ranging from color 1 to color 2 on given image
def radial_gradient(i, poly, p, c1, c2):

    # Draw initial polygon, alpha channel only, on an empty canvas of image size
    ii = Image.new('RGBA', i.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ii)
    draw.polygon(poly, fill=(0, 0, 0, 255), outline=None)

    # Use polygon vertex with highest distance to given point as end of gradient
    p = np.array(p)
    max_dist = max([np.linalg.norm(np.array(v) - p) for v in poly])

    # Calculate color values (gradient) for the whole canvas
    x, y = np.meshgrid(np.arange(i.size[0]), np.arange(i.size[1]))
    c = np.linalg.norm(np.stack((x, y), axis=2) - p, axis=2) / max_dist
    c = np.tile(np.expand_dims(c, axis=2), [1, 1, 3])
    c = (c1 * (1 - c) + c2 * c).astype(np.uint8)
    c = Image.fromarray(c)

    # Paste gradient on temporary image
    ii.paste(c, mask=ii)

    # Paste temporary image on actual image
    i.paste(ii, mask=ii)

    return i


# Create blank canvas with zero alpha channel
w, h = (76*30, 26*30)
image = Image.new('RGBA', (w, h), (0, 0, 0, 0))

# Draw first polygon with radial gradient
polygon = [(0, 0), (76*30, 0),(76*30, 26*30), (0, 26*30), ]
point = (76*30, 26*30)
color1 = (243, 225, 97)
color2 = (229, 196, 44)
image = radial_gradient(image, polygon, point, color1, color2)

# Save image
plt.imshow(image)
plt.show()