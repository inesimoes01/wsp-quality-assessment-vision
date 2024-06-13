from shapely.geometry import Polygon, Point
import numpy as np

# Example contours
contours = [
    [
        [1079, 575], [1077, 577], [1077, 578], [1076, 579], [1076, 581], 
        [1075, 582], [1075, 588], [1076, 589], [1076, 590], [1077, 591], 
        [1077, 592], [1078, 593], [1081, 593], [1082, 592], [1082, 591], 
        [1083, 590], [1083, 589], [1084, 588], [1084, 578], [1083, 577], 
        [1083, 576], [1082, 575]
    ],
    [
        [800, 573], [799, 574], [798, 574], [797, 575], [796, 575], 
        [794, 577], [794, 578], [793, 579], [793, 580], [792, 581], 
        [792, 587], [793, 588], [793, 589], [794, 590], [794, 591], 
        [796, 593], [797, 593], [798, 594], [808, 594], [809, 593], 
        [810, 593], [812, 591], [812, 590], [813, 589], [813, 579], 
        [812, 578], [812, 577], [810, 575], [809, 575], [808, 574], 
        [807, 574], [806, 573]
    ]
]

def contour_to_polygon(contour):
    return Polygon(contour)

def is_contour_inside(contour, other_contour):
    contour_polygon = contour_to_polygon(contour)
    other_polygon = contour_to_polygon(other_contour)
    
    # If the majority of points of one contour are inside another, we consider it inside
    inside_count = sum([other_polygon.contains(Point(pt)) for pt in contour])
    return inside_count > len(contour) / 2

def remove_inside_contours(contours):
    to_remove = set()
    for i, contour in enumerate(contours):
        for j, other_contour in enumerate(contours):
            if i != j and is_contour_inside(contour, other_contour):
                to_remove.add(i)
    
    return [contour for i, contour in enumerate(contours) if i not in to_remove]

# Remove inside contours
filtered_contours = remove_inside_contours(contours)
print(filtered_contours)
