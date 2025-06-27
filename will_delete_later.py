import torch

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
from shapely.affinity   import rotate, scale
import cProfile

device='cuda'

def build_circles(num_circles):
    circle_data = []
    circle_list = []
    circle_coords_list = [] # a mega-list of all the circle coords
    buffer = 1

    for circ_num in range(num_circles):
        radius = int(random.random() * min(xmax, ymax)/15 + 1)
        center_x = random.random() * ((xmax - radius - buffer) - (xmin + radius + buffer)) + xmin + radius + buffer
        center_y = random.random() * ((ymax - radius - buffer) - (ymin + radius + buffer)) + ymin + radius + buffer
        center = (center_x, center_y)

        circle = Point(center_x, center_y).buffer(radius)
        circle_coords = list(circle.exterior.coords)
        circle_coords = [(int(c1), int(c2)) for (c1, c2) in circle_coords] 
        test_data = [center, radius]
        
        # print(f'radius is {radius}, xmax is {xmax}, ymax is {ymax}')

        while (radius < 1 or radius > 7 or c_overlaps(test_data, circle_data) or cr_overlaps()):
            
            radius = int(random.random() * max(xmax, ymax)/15 + 1)
            print(radius)
            center_x = random.random() * ((xmax - radius - buffer) - (xmin + radius + buffer)) + xmin + radius + buffer
            center_y = random.random() * ((ymax - radius - buffer) - (ymin + radius + buffer)) + ymin + radius + buffer
            center = (center_x, center_y)

            circle = Point(center_x, center_y).buffer(radius)
            circle_coords = list(circle.exterior.coords)
            circle_coords = [(int(c1), int(c2)) for (c1, c2) in circle_coords] 
            test_data = [center, radius]

        
        circle_line = LineString(circle.exterior.coords)
        circle_list.append(circle_line)
        circle_coords_list.extend(circle_coords)
        circle_data.append([center, radius])
        print([center, radius])
        # print(f'circle_line is {list(circle.exterior.coords)}')
        # print(circle_coords)

    return circle_list