import numpy as np
import cv2
import copy

def geometric_transforms(self, oo_new_x, oo_new_y, wh_new_x, wh_new_y, w_resize = None, h_resize = None, min_new_area_treshold = 0.7, angle_rotate = 0):
    # oo is the origin of the x and y referencial
    # wh is the oposite diagonal point
    # this method allows for cropping and flipping (both horizontally and vertically) the image by setting appropirate oo_new and wh_new points
    # oo..........wo
    # .  oo_new..  .
    # .  .      .  .
    # .  ..wh_new  .
    # oh..........wh
    # also allows for resizing, by setting w_out and h_out to output image width and height
    # will convert all the annotations accordinglly


    h, w, channels = self.image.shape

    w_crop = np.abs(wh_new_x - oo_new_x)
    h_crop = np.abs(wh_new_y - oo_new_y)

    if(w_resize == None): 
        w_resize = w_crop
    if(h_resize == None):
        h_resize = h_crop

    assert w_crop <= w and w_crop > 0
    assert h_crop <= h and h_crop > 0

    image_new = np.zeros((h_crop, w_crop, channels), dtype=self.image.dtype)

    # determine direction of x and y new, relative to their original direction
    # 1 if the same direction, -1 if oposite direction
    way_x_new = (wh_new_x - oo_new_x) / np.abs(wh_new_x - oo_new_x)
    way_y_new = (wh_new_y - oo_new_y) / np.abs(wh_new_y - oo_new_y)

    assert (way_x_new==1 or way_x_new==-1) and (way_y_new==1 or way_y_new==-1)

    # Map points in original image, into new image referencial
    for y in range(h):
        y_new = int((y - oo_new_y) * way_y_new)
        if 0 <= y_new < h_crop:
            for x in range(w):
                x_new = int((x - oo_new_x) * way_x_new)
                if 0 <= x_new < w_crop:
                    image_new[y_new, x_new, :] = self.image[y, x, :]

    # resize
    scalling_x = w_resize / w_crop
    scalling_y = h_resize / h_crop
    image_new = cv2.resize(image_new, (w_resize, h_resize))
    
    #rotate https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
    image_new, M_rotate = rotate_image(image_new, angle_rotate, border_mode=cv2.BORDER_REPLICATE) # cv2.border replicate to match with the border and better simulate scan


    # update label coordinates
    new_boxes_annotations: list[cvat_annotations_utils.Box_annotation] = []
    new_points_annotations: list[cvat_annotations_utils.Points_annotation] = []
    new_masks_annotations: list[cvat_annotations_utils.Mask_annotation] = []

    # update box_annotations
    for box_annotation in self.image_annotation.boxes_annotation:
        #TODO: no need to deepcopy box_anotation???? SIM É PRECISO!! broo é tipo a quarta vez que aprendes esta lição... 
        box_annotation_copy = copy.deepcopy(box_annotation)
        
        # convert points in original referencial into new referencial
        xtl_new_ref = ((box_annotation_copy.xtl - oo_new_x) * way_x_new) # no need to be int?
        ytl_new_ref = ((box_annotation_copy.ytl - oo_new_y) * way_y_new)
        xbr_new_ref = ((box_annotation_copy.xbr - oo_new_x) * way_x_new)
        ybr_new_ref = ((box_annotation_copy.ybr - oo_new_y) * way_y_new)

        # associate points in new referencial with the points of the actual bounding box. 
        # for example, in a reflection xtl_new_referencial will be the original xtl in the new referencial coords, but with respect of the bounding box, it will be the right most x, ence xbr
        xtl_new = xtl_new_ref if way_x_new == 1 else xbr_new_ref
        xbr_new = xbr_new_ref if way_x_new == 1 else xtl_new_ref
        ytl_new = ytl_new_ref if way_y_new == 1 else ybr_new_ref
        ybr_new = ybr_new_ref if way_y_new == 1 else ytl_new_ref

        # print("xtl: ", box_annotation_copy.xtl, "ytl: ", box_annotation_copy.ytl, "xbr: ", box_annotation_copy.xbr, "ybr: ", box_annotation_copy.ybr, "oo_new_x: ", oo_new_x, "oo_new_y: ", oo_new_y)
        # print("xtl_new: ", xtl_new, "ytl_new: ", ytl_new, "xbr_new: ", xbr_new, "ybr_new: ", ybr_new, "way_x_new: ", way_x_new, "way_y_new: ", way_y_new)

        box_area = (box_annotation_copy.xbr - box_annotation_copy.xtl) * (box_annotation_copy.ybr - box_annotation_copy.ytl) # width * height

        # if at least one corner of bbox inside new image, keep box but only the parts that accuatlly fit inside the new image
        if((0<=xtl_new<w_crop and 0<=ytl_new<h_crop) or (0<=xbr_new<w_crop and 0<=ybr_new<h_crop)):
            # clip values under and over the range, to its minimum and maximum 
            xtl_new = np.clip(xtl_new, 0, w_crop) 
            ytl_new = np.clip(ytl_new, 0, h_crop)
            xbr_new = np.clip(xbr_new, 0, w_crop)
            ybr_new = np.clip(ybr_new, 0, h_crop)

            box_area_new = (xbr_new - xtl_new) * (ybr_new - ytl_new)

            #print("box_area_new: ", box_area_new, " box_area: ", box_area)

            # only keep bbox if significant percentage of its area is kept. it would make sense to keep very small bboxes without any information
            if(box_area_new >= box_area*min_new_area_treshold):
                # have into account the resizing
                xtl_new *= scalling_x
                ytl_new *= scalling_y
                xbr_new *= scalling_x
                ybr_new *= scalling_y

                # have into account rotate
                xtl_new, ytl_new, xbr_new, ybr_new = update_bbox_in_rotated_image(M_rotate, xtl_new, ytl_new, xbr_new, ybr_new)

                # update annoation
                box_annotation_copy.xtl = xtl_new
                box_annotation_copy.ytl = ytl_new
                box_annotation_copy.xbr = xbr_new
                box_annotation_copy.ybr = ybr_new

                new_boxes_annotations.append(box_annotation_copy)

    # update points_annotations
    for points_anotation in self.image_annotation.all_points_annotation:
        points_anotation_copy = copy.deepcopy(points_anotation)

        new_points_coords = []

        for x, y in points_anotation_copy.points_coords:
            # convert into new referencial
            x_new = ((x - oo_new_x) * way_x_new)
            y_new = ((y - oo_new_y) * way_y_new)

            # only keep points inside the new image
            if(0<=x_new<w_crop and 0<=y_new<h_crop):
                # resize
                x_new *= scalling_x
                y_new *= scalling_y
                # rotate
                x_new, y_new = np.dot(M_rotate, (x_new, y_new, 1)).astype(int)

                new_points_coords.append((x_new, y_new))
        
        # only add new annotation if there are points
        if(len(new_points_coords)!=0):
            points_anotation_copy.points_coords = new_points_coords
            new_points_annotations.append(points_anotation_copy)


    # update mask annotations
    for mask_annotation in self.image_annotation.masks_annotation:
        mask_annotation_copy = copy.deepcopy(mask_annotation)
        
        assert self.image_annotation.height == h and self.image_annotation.width == w 
        assert mask_annotation_copy.width <= w

        # print("mask_annotation.width", mask_annotation.width)
        # print("w",  w)
        # print("mask_annotation.top", mask_annotation.top)
        # print("h ", h)
        # Convert cvat rle to binary mask
        mask_image = cvat_annotations_utils.rle_to_binary_image_mask(mask_annotation_copy.rle, mask_annotation_copy.top, mask_annotation_copy.left, mask_annotation_copy.width, self.image_annotation.height, self.image_annotation.width)
        
        # Convert mask image to new coordinates, like before in the new_image
        #TODO: make this a function when i have time

        # here there is no channels argument, since its a single channel binaray mask
        h_mask, w_mask = mask_image.shape
        assert h_mask == h and w_mask == w 

        mask_image_new = np.zeros((h_crop, w_crop), dtype=self.image.dtype)

        mask_num_points_equal_to_1 = np.sum(mask_image == 1)

        for y in range(h_mask):
            y_new = int((y - oo_new_y) * way_y_new)
            if 0 <= y_new < h_crop:
                for x in range(w_mask):
                    x_new = int((x - oo_new_x) * way_x_new)
                    if 0 <= x_new < w_crop:
                        mask_image_new[y_new, x_new] = mask_image[y, x]

        # if mask as at least one value in new image, keep it
        if np.any(mask_image_new == 1):
            new_mask_num_points_equal_to_1 = np.sum(mask_image_new == 1)

            # only keep mask if it paints a significant enough amount of the image, otherwise the mask wouldnt convey enough information
            if(new_mask_num_points_equal_to_1 >= mask_num_points_equal_to_1*min_new_area_treshold):
            
                # resize            
                mask_image_new = cv2.resize(mask_image_new, (w_resize, h_resize))
                # rotate
                mask_image_new, _ = rotate_image(mask_image_new, angle_rotate, border_mode=cv2.BORDER_CONSTANT) # cv2.border_constant to not fill up the border with 1s if the mask touches the border

                assert image_new.shape[:2] == mask_image_new.shape[:2]
                
                # update annotation
                mask_annotation_copy.rle, mask_annotation_copy.top, mask_annotation_copy.left, mask_annotation_copy.width, mask_annotation_copy.height = cvat_annotations_utils.binary_image_mask_to_cvat_mask_rle(mask_image_new)
                new_masks_annotations.append(mask_annotation_copy)

    return image_new, new_boxes_annotations, new_points_annotations, new_masks_annotations

def rotate_image(image, angle_rotate, border_mode=cv2.BORDER_CONSTANT):
    # https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/

    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cx_rotate, cy_rotate) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M_rotate = cv2.getRotationMatrix2D((cx_rotate, cy_rotate), angle_rotate, 1.0)
    cos = np.abs(M_rotate[0, 0])
    sin = np.abs(M_rotate[0, 1])

    # compute the new bounding dimensions of the image
    w_rotate = int((h * sin) + (w * cos))
    h_rotate = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M_rotate[0, 2] += (w_rotate / 2) - cx_rotate
    M_rotate[1, 2] += (h_rotate / 2) - cy_rotate

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M_rotate, (w_rotate, h_rotate), borderMode=border_mode)

    # image = cv2.resize(image, (w,h))
    return image, M_rotate

def update_bbox_in_rotated_image(M_rotate, xtl, ytl, xbr, ybr):

    w = xbr - xtl
    h = ybr - ytl
    assert w > 0 and h > 0

    # calculate remaining corners of bbox
    xtr = xtl + w
    ytr = ybr
    xbl = xtl
    ybl = ytl + h

    # rotate corners of bbox https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html
    corners = [(xtl, ytl), (xtr, ytr), (xbl, ybl), (xbr, ybr)]
    corners_rot = [np.dot(M_rotate, (x, y, 1)).astype(int) for x, y in corners]
    xtl_rot, ytl_rot = corners_rot[0][:2]
    xtr_rot, ytr_rot = corners_rot[1][:2]
    xbl_rot, ybl_rot = corners_rot[2][:2]
    xbr_rot, ybr_rot = corners_rot[3][:2]

    # determine enclosing bounding box which is paralel to the axis of the image containing the rotated image. this is necessary since yolo cant handle bbox which are not paralell to the image axis
    # xtl_new and xbr_new are the min and max of rotated x positions
    # ytl_new, ybr_new are the min and max of rotated y positions
    xtl_new = min(xtl_rot, xtr_rot, xbl_rot, xbr_rot)
    xbr_new = max(xtl_rot, xtr_rot, xbl_rot, xbr_rot)
    ytl_new = min(ytl_rot, ytr_rot, ybl_rot, ybr_rot)
    ybr_new = max(ytl_rot, ytr_rot, ybl_rot, ybr_rot)

    return xtl_new, ytl_new, xbr_new, ybr_new
