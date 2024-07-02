import numpy as np
import cv2
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import copy
import math
from PIL.PngImagePlugin import PngInfo
import random
import CreateDroplet
import CreateBackground
from DropletShape import DropletShape
from shapely import union
from shapely import Polygon
import ShapeList
from shapely import geometry
from shapely.ops import unary_union
import RosinRammlerDistribution
import CreateMask


sys.path.insert(0, 'src/common')
import config as config
from Util import *

sys.path.insert(0, 'src')
from Droplet import *
from CreateColors import *
from shapely.geometry import MultiPolygon, Polygon

class ShapeInImage:
    def __init__(self, shape_index:int, points, droplet_data:Droplet):
        self.shape_index = shape_index
        self.points = points
        self.droplet_data = droplet_data


class CreateWSP:
    def __init__(self, index:int, colors:Colors, shapes:list[DropletShape], polygons_by_size, num_spots, type_of_dataset):
        self.filename = index
        self.max_num_spots:int = config.MAX_NUM_SPOTS
        self.min_num_spots:int = config.MIN_NUM_SPOTS
        self.width:float = config.WIDTH_MM * config.RESOLUTION
        self.height:float = config.HEIGHT_MM * config.RESOLUTION
        self.num_spots = num_spots
        self.colors = colors
        self.polygons_by_size = polygons_by_size
        self.shapes = shapes
        self.droplet_area = 0
        self.type_of_dataset = type_of_dataset


        # generate the droplet sizes, which is assumed to be the area, given a rosin rammler distribution and the centers
        self.droplet_area_rosin = RosinRammlerDistribution.generate_droplet_sizes_rosin_rammler(self.num_spots)

        # save latex files
        if index == 0:
            RosinRammlerDistribution.fit_droplet_size_graph(self.droplet_area_rosin)
        
        # generate the center coordinates for each one of the droplets
        self.droplet_coordinates = self.generate_droplet_coordinates()

        # create the wsp background
        # background colors change to create a more diverse dataset
       
        CreateBackground.create_background(self.colors.background_colors, self.width, self.height)
        self.rectangle = cv2.imread(config.DATA_ARTIFICIAL_RAW_BACKGROUND_IMG, cv2.IMREAD_COLOR)
        self.rectangle = cv2.cvtColor(self.rectangle, cv2.COLOR_BGR2RGB)
        

        # draw and color the droplets on the background given the stats created before
        self.generate_one_wsp()
        self.save_image()


    def generate_one_wsp(self):
        self.list_of_individual_shapes_in_image:list[ShapeInImage]  = []
        self.list_of_intersected_shapes_in_image = []
        self.list_of_singular_shapes_in_image = []
        self.annotation_labels = []

       
        self.droplets_data:list[Droplet] = []

        match self.type_of_dataset:

            case 0:     # only singular circles inside the wsp
                for i in range(self.num_spots):
                    spot_radius = math.ceil(self.droplet_area_rosin[i])
                    
                    count = 0
                    while (True):
                        center_x = np.random.randint(spot_radius, self.width - spot_radius)
                        center_y = np.random.randint(spot_radius, self.height - spot_radius)

                        overlap_bool = self.check_if_overlapping(center_x, center_y, spot_radius, i)
                
                        if not overlap_bool:
                            self.save_draw_droplet(int(center_x), int(center_y), int(spot_radius), spot_color, i)
                            break
                        
                        count += 1
                        if count > 100:
                            break
            
                for i in range(self.num_spots):
                    spot_radius = math.ceil(self.droplet_area_rosin[i])
                    
                    count = 0
                    while (True):
                        center_x = np.random.randint(spot_radius, self.width - spot_radius)
                        center_y = np.random.randint(spot_radius, self.height - spot_radius)

                        overlap_bool = self.check_if_overlapping(center_x, center_y, spot_radius, i)
                
                        if not overlap_bool:
                            self.save_draw_droplet(int(center_x), int(center_y), int(spot_radius), spot_color, i)
                            break
                        
                        count += 1
                        if count > 100:
                            break
                    
            case 1:     # overlapped and singular droplets (circles and elipses)
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
                #for i in range(4):
                    # get pre generated values
                    spot_area = math.ceil(self.droplet_area_rosin[i])
                    center_x, center_y = self.droplet_coordinates[i]
                    
                    # save informations for each individual droplet
                    shape_to_save = self.save_individual_droplets(int(center_x), int(center_y), int(spot_area), i)
                    
                    # accumulate the labels for each polygon for yolo training
                    self.annotation_labels.append(CreateMask.polygon_to_yolo_label(shape_to_save.points, self.width, self.height))

                    # save information of sliding window from before and after
                    if sliding_window_x < center_x < sliding_window_x + sliding_window_threshold:
                        shapes_in_window_threshold_after.append(shape_to_save)
                    else:
                        shapes_in_window.append(shape_to_save)

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

                        # accumulate the area of each polygon for statistics
                        self.droplet_area += polygon.area

                        self.list_of_singular_shapes_in_image.append(polygon)
                        CreateDroplet.paint_polygon(self.rectangle, polygon)
 
                # paint the overlapped droplets
                for polygon in self.list_of_intersected_shapes_in_image:
                    # accumulate the area of each polygon for statistics
                    self.droplet_area += polygon.area

                    CreateDroplet.paint_polygon(self.rectangle, polygon)
                    

            case 2:     # classes overlapped and singular are leveled
                count_overlapping_sets = 0
                count_single = 0
                num_overlapping_droplets = math.ceil(self.num_spots * 1 / 3)  
                num_overlapping_sets = math.ceil(self.num_spots * 1 / 3) 
                overlapped_sets = self.generate_overlapping_value(num_overlapping_sets, num_overlapping_droplets)
                sumy = sum(overlapped_sets)
                count_total_no_droplets = 0
                
                while(self.num_spots > count_total_no_droplets):
                
                    # create only overlapping circles
                    if count_overlapping_sets < num_overlapping_sets:
                        no_drops_in_set = overlapped_sets[count_overlapping_sets]

                        # initial circle position
                        center_x = np.random.randint(5, self.width - 5)
                        center_y = np.random.randint(5, self.height - 5)

                        for k in range(no_drops_in_set): 
                            spot_radius = math.ceil(self.droplet_area_rosin[count_total_no_droplets])
                            no_rand = random.randint(0, 1)
                            # if n_spot is even it will move to be on the right of the original circle
                            # if n_spot is odd it will move to be under the original circle
                            if no_rand == 0:
                                center_x += self.droplet_area_rosin[count_total_no_droplets - 1] + self.droplet_area_rosin[count_total_no_droplets]
                            else: 
                                center_y += self.droplet_area_rosin[count_total_no_droplets - 1] + self.droplet_area_rosin[count_total_no_droplets]
                            # if k % 2 == 0: center_x += self.droplet_area_rosin[count_total_no_droplets - 1 ]
                            # else: center_y += self.droplet_area_rosin[count_total_no_droplets - 1] 
                                
                            count_total_no_droplets += 1

                            self.save_draw_droplet(int(center_x), int(center_y), int(spot_radius), spot_color, count_total_no_droplets)

                        count_overlapping_sets += 1
                    
                    # create only single droplets
                    else:
                        spot_radius = math.ceil(self.droplet_area_rosin[count_total_no_droplets])
                        if spot_radius < config.DROPLET_COLOR_THRESHOLD: spot_color = self.droplet_color_big[np.random.randint(0, len(self.droplet_color_big))]
                        else: spot_color = self.droplet_color_small[np.random.randint(0, len(self.droplet_color_small))]

                        while (True):
                            center_x = np.random.randint(spot_radius, self.width - spot_radius)
                            center_y = np.random.randint(spot_radius, self.height - spot_radius)

                            overlapping = self.check_if_overlapping(center_x, center_y, spot_radius, count_total_no_droplets)

                            if not overlapping:
                                break
                        
                        count_total_no_droplets += 1
                        self.save_draw_droplet(int(center_x), int(center_y), int(spot_radius), spot_color, count_total_no_droplets)


    def generate_droplet_coordinates(self):
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
        
    def save_individual_droplets(self, center_x, center_y, spot_area, index_droplet):
        index_droplet += 1

        # choose shape for droplet based on the area
        shape_id = CreateDroplet.choose_polygon(self.polygons_by_size, spot_area)
        curr_shape = self.shapes[shape_id]

        # get the position of each coordinate given the center the polygon should take place
        translated_points = CreateDroplet.get_shape_translation(curr_shape, center_x, center_y)

        shape_to_save = ShapeInImage(shape_id, translated_points, Droplet(center_x, center_y, spot_area, index_droplet, []))
        self.list_of_individual_shapes_in_image.append(shape_to_save)

        return shape_to_save

    def generate_overlapping_value(self, no_overlap_sets, no_droplets):
        numbers = [2] * no_overlap_sets
        remaining_sum = no_droplets - no_overlap_sets* 2

        while remaining_sum > 0:
            index = random.randint(0, no_overlap_sets - 1)
            numbers[index] += 1
            remaining_sum -= 1

        return numbers
    
    def check_if_overlapping(self, center_x, center_y, spot_radius, i, areThereElipses):
        overlapping = False

        for droplet in self.droplets_data:
            distance = math.sqrt((center_x - droplet.center_x)**2 + (center_y - droplet.center_y)**2)

            # circle and circle
            if ((i % 10 == 1) and not droplet.isElipse):
                # add a small value to make sure the droplets are actually overlapped and not just touching
                if distance < spot_radius + droplet.radius + config.OVERLAPPING_THRESHOLD:
                    overlapping = True
                    break

            # circle and elipse or elipse and circle
            elif (((i % 10 == 1) and droplet.isElipse) or ((i % 10 == 0) and not droplet.isElipse)) and areThereElipses:
                if distance < spot_radius + droplet.radius + config.ELIPSE_MAJOR_AXE_VALUE + config.OVERLAPPING_THRESHOLD:
                    overlapping = True
                    break

            # elipse and elipse
            elif ((i % 10 == 0) and droplet.isElipse) and areThereElipses:
                if distance < spot_radius + droplet.radius + config.ELIPSE_MAJOR_AXE_VALUE * 2 + config.OVERLAPPING_THRESHOLD:
                    overlapping = True
                    break

        return overlapping
    
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
        path = os.path.join(config.DATA_ARTIFICIAL_RAW_IMAGE_DIR, str(self.filename) + '.png')

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
