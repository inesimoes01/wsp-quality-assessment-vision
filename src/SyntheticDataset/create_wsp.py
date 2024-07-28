import numpy as np
import cv2
import sys
import os
import numpy as np
import copy
import math
import random
import matplotlib.pyplot as plt

from shapely import geometry
from shapely import Polygon
from shapely.ops import unary_union



import config as config
import rosinrammler_distribution as rosinrammler_distribution
import create_masks as create_masks
import create_droplets as create_droplets
import create_background as create_background
from create_colors import Colors
from droplet_shape import DropletShape
sys.path.insert(0, 'src/common')
from Droplet import Droplet 


class ShapeInImage:
    def __init__(self, shape_index:int, points, droplet_data:Droplet):
        self.shape_index = shape_index
        self.points = points
        self.droplet_data = droplet_data

class CreateWSP:
    def __init__(self, index:int, colors:Colors, shapes:list[DropletShape], polygons_by_size, num_spots, type_of_dataset, image_resolution, characteristic_particle_size, uniform):
        self.filename = index
        self.max_num_spots:int = config.MAX_NUM_SPOTS
        self.min_num_spots:int = config.MIN_NUM_SPOTS
        self.width:int = int(config.WIDTH_MM * image_resolution)
        self.height:int = int(config.HEIGHT_MM * image_resolution)
        self.num_spots = num_spots
        self.colors = colors
        self.polygons_by_size = polygons_by_size
        self.shapes = shapes
        self.droplet_area = 0
        self.type_of_dataset = type_of_dataset

        # generate the droplet sizes, which is assumed to be the area, given a rosin rammler distribution and the centers
        self.droplet_size_rosin = rosinrammler_distribution.generate_droplet_sizes_rosin_rammler(self.num_spots, characteristic_particle_size, uniform)

        # save latex files
        if index == 0:
            rosinrammler_distribution.fit_droplet_size_graph(self.droplet_size_rosin)

        # create the wsp background
        # background colors change to create a more diverse dataset
        create_background.create_background(self.colors.background_colors, self.width, self.height)
        self.rectangle = cv2.imread(config.DATA_SYNTHETIC_WSP_BACKGROUND_IMG, cv2.IMREAD_COLOR)
        self.rectangle = cv2.cvtColor(self.rectangle, cv2.COLOR_BGR2RGB)

        # draw and color the droplets on the background given the stats created before
        self.generate_one_wsp()

        # save the 
        self.save_image()


    def generate_one_wsp(self):
        # individual are all the shapes that were placed in the image with the center identified
        self.list_of_individual_shapes_in_image:list[ShapeInImage]  = []

        # intersected is all the joint polygons points and singular the polygons that did not intersect
        # these two lists have all the polygon shapes that are represented in the wsp
        self.list_of_intersected_shapes_in_image = []
        self.list_of_singular_shapes_in_image = []

        self.annotation_labels = []       
        self.droplets_data:list[Droplet] = []

        match self.type_of_dataset:

            case 0:     # only singular shapes inside the wsp

                self.overlapped_dictionary = {}
                self.polygon_dictionary = {}

                # draw the outline of all the droplets with the generated radius and center
                # using a sliding window to be less computationally heavy when searching for overlappig droplets
                for i in range(self.num_spots):
              
                    # get pre generated values for the radius
                    spot_size = math.ceil(self.droplet_size_rosin[i])
                    
                    # choose the shape to put in the image
                    shape_id, shape_to_save = self.choose_shape(spot_size)

                    # based on the size of the shape, find a coordinate for the center where the shape does not meet any of the other already placed shapes
                    center_x, center_y, translated_points = self.generate_droplet_coordinates_singular(shape_to_save, spot_size)
                    shape_to_draw = self.save_individual_droplets(shape_to_save, shape_id, translated_points, int(center_x), int(center_y), int(spot_size), i)
                    
                    # accumulate the labels for each polygon for yolo training
                    self.annotation_labels.append(create_masks.polygon_to_yolo_label(shape_to_draw.points, self.width, self.height))

                # paint the individual shapes in the image
                for shape in self.list_of_individual_shapes_in_image:
                    if shape.droplet_data.overlappedIDs == []:
                        polygon = geometry.Polygon(shape.points)

                        self.list_of_singular_shapes_in_image.append(polygon)
                        create_droplets.paint_polygon(self.rectangle, polygon, self.colors.light_color, self.colors.dark_rgb, self.colors.outside_color)
 

            case 1:     # overlapped and singular droplets
                # generate the center coordinates for each one of the droplets
                self.droplet_coordinates = self.generate_droplet_coordinates(self.type_of_dataset)

                sliding_window_x = 20
                sliding_window_threshold = 10
                shapes_in_window = []
                shapes_in_window_threshold_before = []
                shapes_in_window_threshold_after = []

                self.overlapped_dictionary = {}
                self.polygon_dictionary = {}

                # draw the outline of all the droplets with the generated radius and center
                # using a sliding window to be less computationally heavy when searching for overlappig droplets
                for i in range(self.num_spots):
              
                    # get pre generated values
                    spot_size = math.ceil(self.droplet_size_rosin[i])
                    center_x, center_y = self.droplet_coordinates[i]

                    # save informations for each individual droplet
                    shape_id, shape_to_save = self.choose_shape(spot_size)
                    translated_points = create_droplets.get_shape_translation(shape_to_save, center_x, center_y)
                    shape_to_draw = self.save_individual_droplets(shape_to_save, shape_id, translated_points, int(center_x), int(center_y), int(spot_size), i)
                    
                    # accumulate the labels for each polygon for yolo training
                    self.annotation_labels.append(create_masks.polygon_to_yolo_label(shape_to_draw.points, self.width, self.height))

                    # save information of sliding window from before and after
                    if sliding_window_x < center_x < sliding_window_x + sliding_window_threshold:
                        shapes_in_window_threshold_after.append(shape_to_draw)
                    else:
                        shapes_in_window.append(shape_to_draw)

                    # when window is over, check intersections of the shapes in that area with the before and after sliding window
                    if center_x > sliding_window_x + sliding_window_threshold:
                        shapes_in_window.extend(shapes_in_window_threshold_before)
                        shapes_in_window.extend(shapes_in_window_threshold_after)
                        self.merge_intersections(shapes_in_window)
                        
                        shapes_in_window_threshold_before = shapes_in_window_threshold_after
                        shapes_in_window_threshold_after = []
                        shapes_in_window = []

                        sliding_window_x += sliding_window_x
                
                # save the final shapes from the last pixels and check intersections
                shapes_in_window.extend(shapes_in_window_threshold_before)
                shapes_in_window.extend(shapes_in_window_threshold_after)
                self.merge_intersections(shapes_in_window)

                # based on the overlapping ids, find the groups of droplets that are overlapped
                components = self.find_connected_components(self.overlapped_dictionary)
                for component in components:
                    union_poly = unary_union([self.polygon_dictionary[node] for node in component])
                    self.list_of_intersected_shapes_in_image.append(union_poly)

                # paint the individual shapes in the image
                for shape in self.list_of_individual_shapes_in_image:
                    if shape.droplet_data.overlappedIDs == []:
                        polygon = geometry.Polygon(shape.points)
                        self.list_of_singular_shapes_in_image.append(polygon)
                        create_droplets.paint_polygon(self.rectangle, polygon, self.colors.light_color, self.colors.dark_rgb, self.colors.outside_color)
 
                # paint the overlapped droplets
                for polygon in self.list_of_intersected_shapes_in_image:
                    create_droplets.paint_polygon(self.rectangle, polygon, self.colors.light_color, self.colors.dark_rgb, self.colors.outside_color)


    def generate_droplet_coordinates_singular(self, shape_to_save, spot_size):
       
        while (True):
            overlapping = False
            center_x = np.random.randint(spot_size, self.width - spot_size)
            center_y = np.random.randint(spot_size, self.height - spot_size)
            
            # get the position of each coordinate given the center the polygon should take place
            translated_points = create_droplets.get_shape_translation(shape_to_save, center_x, center_y)
            polygon_to_check = Polygon(translated_points)

            for droplet in self.list_of_individual_shapes_in_image:
                aux_drop_poly = geometry.Polygon(droplet.points)

                if polygon_to_check.intersects(aux_drop_poly) and not polygon_to_check.touches(aux_drop_poly):
                    overlapping = True
                    break

            if not overlapping: break

        return center_x, center_y, translated_points
                    


    def generate_droplet_coordinates(self, type_of_dataset):
        droplet_coordinates_x = np.random.randint(0, self.width, self.num_spots)
        droplet_coordinates_y = np.random.randint(0, self.height, self.num_spots)
        droplet_coordinates = np.column_stack((droplet_coordinates_x, droplet_coordinates_y)) 
        sorted_indices = np.lexsort((droplet_coordinates[:, 1], droplet_coordinates[:, 0]))
        
        return droplet_coordinates[sorted_indices]
    
    def merge_intersections(self, shapes_in_window:list[ShapeInImage]):

        # saves all the overlapped ids
        for i, curr_drop in enumerate(shapes_in_window):
            curr_drop_poly = geometry.Polygon(curr_drop.points)
            
            # associates the droplet id with the polygon
            self.polygon_dictionary[curr_drop.droplet_data.id] = curr_drop_poly
            
            for j in range(i + 1, len(shapes_in_window)):
                aux_drop = shapes_in_window[j]
                aux_drop_poly = geometry.Polygon(aux_drop.points)
                
                if curr_drop_poly.intersects(aux_drop_poly) and not curr_drop_poly.touches(aux_drop_poly):
                    
                    if curr_drop.droplet_data.id not in aux_drop.droplet_data.overlappedIDs and curr_drop.droplet_data.id != aux_drop.droplet_data.id:
                        curr_drop.droplet_data.overlappedIDs += [aux_drop.droplet_data.id]
                        aux_drop.droplet_data.overlappedIDs += [curr_drop.droplet_data.id]

                        if curr_drop.droplet_data.id not in self.overlapped_dictionary:
                            self.overlapped_dictionary[curr_drop.droplet_data.id] = []

                        if aux_drop.droplet_data.id not in self.overlapped_dictionary:
                            self.overlapped_dictionary[aux_drop.droplet_data.id] = []

                        self.overlapped_dictionary[curr_drop.droplet_data.id].append(aux_drop.droplet_data.id)
                        self.overlapped_dictionary[aux_drop.droplet_data.id].append(curr_drop.droplet_data.id)        
    
    def find_connected_components(self, overlap_dict):
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in overlap_dict[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in overlap_dict:
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    def choose_shape(self, spot_size):
        # choose shape for droplet based on the area
        shape_id = create_droplets.choose_polygon(self.polygons_by_size, spot_size)
        curr_shape = self.shapes[shape_id]

        return shape_id, curr_shape
        
    def save_individual_droplets(self, shape_to_save, shape_id, translated_points, center_x, center_y, spot_size, index_droplet):
        index_droplet += 1
        
        # calculate droplet area and tranlation points
        pol = Polygon(shape_to_save.roi_points)
        spot_area = pol.area
        self.droplet_area += spot_area
    

        shape = ShapeInImage(shape_id, translated_points, Droplet(center_x, center_y, spot_area, index_droplet, [], spot_size) )
        self.list_of_individual_shapes_in_image.append(shape)

        return shape

    def generate_overlapping_value(self, no_overlap_sets, no_droplets):
        numbers = [2] * no_overlap_sets
        remaining_sum = no_droplets - no_overlap_sets* 2

        while remaining_sum > 0:
            index = random.randint(0, no_overlap_sets - 1)
            numbers[index] += 1
            remaining_sum -= 1

        return numbers
    
    # def check_if_overlapping(self, center_x, center_y, spot_radius, i, areThereElipses):
    #     overlapping = False

    #     for droplet in self.droplets_data:
    #         distance = math.sqrt((center_x - droplet.center_x)**2 + (center_y - droplet.center_y)**2)

    #         # circle and circle
    #         if ((i % 10 == 1) and not droplet.isElipse):
    #             # add a small value to make sure the droplets are actually overlapped and not just touching
    #             if distance < spot_radius + droplet.radius + config.OVERLAPPING_THRESHOLD:
    #                 overlapping = True
    #                 break

    #         # circle and elipse or elipse and circle
    #         elif (((i % 10 == 1) and droplet.isElipse) or ((i % 10 == 0) and not droplet.isElipse)) and areThereElipses:
    #             if distance < spot_radius + droplet.radius + config.ELIPSE_MAJOR_AXE_VALUE + config.OVERLAPPING_THRESHOLD:
    #                 overlapping = True
    #                 break

    #         # elipse and elipse
    #         elif ((i % 10 == 0) and droplet.isElipse) and areThereElipses:
    #             if distance < spot_radius + droplet.radius + config.ELIPSE_MAJOR_AXE_VALUE * 2 + config.OVERLAPPING_THRESHOLD:
    #                 overlapping = True
    #                 break

    #     return overlapping
    
    def add_shadow(self):
        # create shadow shape
        shadow_mask = np.zeros_like(self.rectangle[:, :, 0], dtype=np.uint8)
        rows, cols = shadow_mask.shape
        triangle = np.array([[0, rows], [cols, rows], [0, 0]], np.int32)
        cv2.fillPoly(shadow_mask, [triangle], 255)
        shadow_mask_3channel = cv2.merge((shadow_mask, shadow_mask, shadow_mask))

        # apply shadow effect 
        return cv2.addWeighted(self.rectangle, 1, shadow_mask_3channel, -0.2, -0.5)

    def save_image(self):
        path = os.path.join(config.DATA_SYNTHETIC_NORMAL_WSP_DIR, config.DATA_GENERAL_IMAGE_FOLDER_NAME, str(self.filename) + '.png')

        self.blur_image = cv2.GaussianBlur(self.rectangle, (3, 3), 0)
        rgb_image = cv2.cvtColor(self.blur_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, rgb_image)



 
        # combined_polygons = []
        # for i, shape1 in enumerate(list_shapes_image_to_check):
        #     for j, shape2 in enumerate(list_shapes_image_to_check, i+1):
        #         if geometry.Polygon(shape1.points).within(geometry.Polygon(shape2.points)):
        #             # save it to the list
                    





        # # iterate over each polygon and compare it to all polygons, collecting the whole intersection
        # while list_shapes_image_to_check:
        #     curr_shape_dict = list_shapes_image_to_check.pop(0)
        #     curr_points = curr_shape_dict.points
        #     base_polygon = geometry.Polygon(curr_points)
        #     intersected = False
        #     combined_polygon = base_polygon
            


        #     while polygons:
        #         polygon = polygons.pop(0)
        #         if combined_polygon.within(polygon):
 
        #             # join shapes and save it to the list
        #             combined_polygon = union(combined_polygon, polygon)
        #             polygons.append(combined_polygon)

        #             intersected = True
                    

        #     if intersected and not isinstance(combined_polygon, MultiPolygon):
        #         #combined_polygon = union(intersected)

        #         if not combined_polygon.is_empty:
        #             # add new shape to the original shape list with the roi points
        #             new_shape = ShapeList.create_new_shape_overlapped(id_counter, combined_polygon)
        #             self.shapes.append(new_shape)
                    
        #             # add new polygon to the final list of polygons in the image
        #             points = list(combined_polygon.exterior.coords)
        #             points = np.array(points, np.int32)

        #             self.list_of_intersected_shapes_in_image.append(ShapeInImage(id_counter, points))
        #             id_counter += 1

        #     else:
        #         self.list_of_intersected_shapes_in_image.append(ShapeInImage(curr_shape_dict.index, curr_points))
