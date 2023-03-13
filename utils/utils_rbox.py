'''
Author: [egrt]
Date: 2023-01-30 19:00:28
LastEditors: Egrt
LastEditTime: 2023-03-13 16:22:48
Description: Oriented Bounding Boxes utils
'''

import numpy as np
import math
pi = np.pi 
import cv2
import torch

def poly2rbox(polys):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    Returns:
        rboxes (array): (num_gts, [cx cy l s θ]) 
    """
    assert polys.shape[-1] == 8
    rboxes = []
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        theta = angle / 180 * pi # 转为pi制
        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while not np.pi / 2 > theta >= -np.pi / 2:
            if theta >= np.pi / 2:
                theta -= np.pi
            else:
                theta += np.pi
        assert np.pi / 2 > theta >= -np.pi / 2
        rboxes.append([x, y, w, h, theta])
    return np.array(rboxes)

def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a
    
def poly2hbb(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly) 

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h]) 
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1)
    else:
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1) 
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=1)
    return hbboxes

def rbox2poly(obboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    try:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    except:
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    polys = get_best_begin_point(polys)
    return polys

def cal_line_length(point1, point2):
    """Calculate the length of line.
    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]
    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.
    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8)))


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.
    Args:
        coordinate (ndarray): shape(n, 8).
    Returns:
        reorder coordinate (ndarray): shape(n, 8).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def correct_rboxes(rboxes, image_shape):
    """将polys按比例进行缩放
    Args:
        coordinate (ndarray): shape(n, 8).
    Returns:
        reorder coordinate (ndarray): shape(n, 8).
    """
    polys = rbox2poly(rboxes)
    nh, nw = image_shape
    polys[:, [0, 2, 4, 6]] *= nw
    polys[:, [1, 3, 5, 7]] *= nh

    return polys
