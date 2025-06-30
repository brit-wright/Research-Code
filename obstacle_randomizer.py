### want to add random obstacles to the map

# let's define some constraints for adding obstacles
"""
let's start with rectangles because rectangles are easy lol
a rectangle would be defined by four vertices. let's all them tr, tl, br, bl
tr: top right
tl: top left
br: bottom right
bl: bottom left

i think these vertices should be tuples [x, y]

Constraints:

tr[1] = tl[1]
br[1] = bl[1]

tr[0] = br[0]
tl[0] = bl[0]

thus we define everything be x1, y1, x2, y2 (using some form of random number generator)

constraints on the random number generator:

(10 <= abs(x1-x2) and 10 <= abs(y1-y2)) and (abs(x1-x2) <= 15 or abs(y1-y2) <= 15)

Then we set 
min(x1, x2) = tl[0] = bl[0]
max(x1, x2) = tr[0] = br[0]
min(y1, y2) = br[1] = bl[1]
max(y1, y2) = tl[1] = tr[1]

Let's try it out
"""

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


########################## OUTSIDE WALLS ########################################

x1 = 0
x2 = int(random.random() * 100)
y1 = 0
y2 = int(random.random() * 100)

# print(x1, x2, y1, y2)
# print(abs(x1 - x2))
# print(abs(y1 - y2))

while ((abs(x1 - x2) <= 40 or abs(y1 - y2) <= 40) or (abs(x1 - x2) > 60 or abs(y1 - y2) > 60)):
    x2 = int(random.random() * 100)
    y2 = int(random.random() * 100)

# print(x1, x2, y1, y2)
# print(abs(x1 - x2))
# print(abs(y1 - y2))


xmin = min(x1, x2)
xmax = max(x1, x2)
ymin = min(y1, y2)
ymax = max(y1, y2)


(xA, xB, xC) = ( 5, 12, 15)
(yA, yB, yC)     = ( 5, 10, 12)

# xlabels = (xmin, xA, xB, xC, xmax)
xlabels = list(range(xmin, xmax+1, 2))
# ylabels = (ymin, yA, yB, yC, ymax)
ylabels = list(range(ymin, ymax+1, 2))

tl = [xmin, ymax]
tr = [xmax, ymax]
bl = [xmin, ymin]
br = [xmax, ymin]

# print(tl, tr, bl, br)

########################## INNER OBSTACLES #########################

# first, let's just get practive with drawing obstacles 

# smaller squares/rectangles:

# corner would have shape [x, y] and the sides would have shape [[x, y], [x, y]]
    
def rectangles_overlap(min_x, min_y, max_x, max_y, existing_min_x, existing_min_y, existing_max_x, existing_max_y):

    # check for rectangle overlap by 

    # if anything in the parentheses is True, then it means overall that there is no overlap
    # so then rectangles_overlap is false
    return not (max_x <= existing_min_x or min_x >= existing_max_x or max_y <= existing_min_y or min_y >= existing_max_y)

def r_overlaps(rx1, rx2, ry1, ry2, rtr_list, rtl_list, rbr_list, rbl_list):
    
    if rtr_list == []:
        return False

    min_x = min(rx1, rx2)
    max_x = max(rx1, rx2)
    min_y = min(ry1, ry2)
    max_y = max(ry1, ry2)
 
    for i in range(len(rtr_list)):
        clearance = 3
        existing_min_x = min(rtl_list[i][0], rbl_list[i][0]) - clearance
        existing_max_x = max(rtr_list[i][0], rbr_list[i][0]) + clearance
        existing_min_y = min(rbl_list[i][1], rbr_list[i][1]) - clearance
        existing_max_y = max(rtl_list[i][1], rtr_list[i][1]) + clearance

        if rectangles_overlap(min_x, min_y, max_x, max_y, existing_min_x, existing_min_y, existing_max_x, existing_max_y):
            return True

    return False

def build_rectangles(num_rectangles):
    rectangle_list = []
    rectangle_coords_list = []
    rbr_list = []
    rbl_list = []
    rtl_list = []
    rtr_list = []
    for rect_num in range(num_rectangles):

        rx1 = int(random.random() * (xmax - xmin) + xmin)
        rx2 = int(random.random() * (xmax - xmin) + xmin)
        ry1 = int(random.random() * (ymax - ymin) + ymin)
        ry2 = int(random.random() * (ymax - ymin) + ymin)

        # print(xmax, ymax)
        while ((abs(rx1 - rx2) <= 2 or abs(ry1 - ry2) <= 2) or (abs(rx1 - rx2) > 6 or abs(ry1 - ry2) > 6) 
            or (min(rx1, rx2) < xmin) or (max(rx1, rx2) > xmax) or (min(ry1, ry2) < ymin) or (max(ry1, ry2) > ymax) or r_overlaps(rx1, rx2, ry1, ry2, rtr_list, rtl_list, rbr_list, rbl_list)):
            rx1 = int(random.random() * (xmax - xmin) + xmin)
            rx2 = int(random.random() * (xmax - xmin) + xmin)
            ry1 = int(random.random() * (ymax - ymin) + ymin)
            ry2 = int(random.random() * (ymax - ymin) + ymin)

        rxmin = min(rx1, rx2)
        rxmax = max(rx1, rx2)
        rymin = min(ry1, ry2)
        rymax = max(ry1, ry2)

        rtl = [rxmin, rymax]
        rtr = [rxmax, rymax]
        rbl = [rxmin, rymin]
        rbr = [rxmax, rymin]

        rectangle = LineString([rbl, rbr, rtr, rtl, rbl])

        rbl_list.append(rbl)
        rbr_list.append(rbr)
        rtr_list.append(rtr)
        rtl_list.append(rtl)

        rectangle_list.append(rectangle)

    # create a coordinates list for the sides of the rectangle
    # start with the top line
    # print(f'length of rb_list is {len(rbl_list)}')
    for r_index in range(len(rbl_list)):
        rectangle_coords = []
        # find the coordinates of the top and bottom lines
        rtx = rtl_list[r_index][0] # the x-coordinate of the top-left line

        while rtx != rtr_list[r_index][0] + 1:
            rectangle_coords.append((rtx, rtl_list[r_index][1]))
            rectangle_coords.append((rtx, rbl_list[r_index][1]))
            rtx += 1

        # find the coordinates of the left and right lines
        rty = rbl_list[r_index][1] # the y-coordinate of the bottom-left line

        while rty != rtl_list[r_index][1] + 1:
            rectangle_coords.append((rtl_list[r_index][0], rty))
            rectangle_coords.append((rtr_list[r_index][0], rty))
            rty += 1

        rectangle_coords_list.append(rectangle_coords)

    return rectangle_list, rectangle_coords_list

def c_overlaps(test_data, circle_data):

    for circle in circle_data:

        distance = sqrt((test_data[0][0] - circle[0][0])**2 + (test_data[0][1] - circle[0][1])**2)
        distance_threshold = test_data[1] + circle[1]

        if distance < distance_threshold - 1:
            return True

    return False

def build_circles(num_circles):
    circle_data = []
    circle_list = []
    circle_coords_list = [] # a mega-list of all the circle coords
    buffer = 1

    for circ_num in range(num_circles):
        radius = int(random.random() * min(xmax, ymax)/25 + 1)
        center_x = random.random() * ((xmax - radius - buffer) - (xmin + radius + buffer)) + xmin + radius + buffer
        center_y = random.random() * ((ymax - radius - buffer) - (ymin + radius + buffer)) + ymin + radius + buffer
        center = (center_x, center_y)

        circle = Point(center_x, center_y).buffer(radius)
        circle_coords = list(circle.exterior.coords)
        circle_coords = [(int(c1), int(c2)) for (c1, c2) in circle_coords] 
        test_data = [center, radius]
        
        # print(f'radius is {radius}, xmax is {xmax}, ymax is {ymax}')

        while (radius < 1 or radius > 7 or c_overlaps(test_data, circle_data)):
            
            radius = int(random.random() * max(xmax, ymax)/15 + 1)
            # print(radius)
            center_x = random.random() * ((xmax - radius - buffer) - (xmin + radius + buffer)) + xmin + radius + buffer
            center_y = random.random() * ((ymax - radius - buffer) - (ymin + radius + buffer)) + ymin + radius + buffer
            center = (center_x, center_y)

            circle = Point(center_x, center_y).buffer(radius)
            circle_coords = list(circle.exterior.coords)
            circle_coords = [(int(c1), int(c2)) for (c1, c2) in circle_coords] 
            test_data = [center, radius]

        
        circle_line = LineString(circle.exterior.coords)
        circle_list.append(circle_line)
        circle_coords_list.append(circle_coords)
        circle_data.append([center, radius])
        # print([center, radius])
        # print(f'circle_line is {list(circle.exterior.coords)}')
        # print(circle_coords)

    return circle_list, circle_coords_list

# using the bounding box method
def ell_overlaps(ellipse, ellipse_data):
    
    if ellipse_data == []:
        return False
    
    tol = 6
    # the ellipse 'data type' is of the form [center, (x_axis, y_axis), angle]

    # create the bounding box for the test_ellipse
    test_tl = [ellipse[0][0] - ellipse[1][0]  , ellipse[0][1] + ellipse[1][1]  ]
    test_tr = [ellipse[0][0] + ellipse[1][0]  , ellipse[0][1] + ellipse[1][1]  ]
    test_bl = [ellipse[0][0] - ellipse[1][0]  , ellipse[0][1] - ellipse[1][1]  ]
    test_br = [ellipse[0][0] + ellipse[1][0]  , ellipse[0][1] - ellipse[1][1]  ]

    for ell in ellipse_data:
        # add a buffer
        ell_test_tl = [ell[0][0] - ell[1][0] - tol , ell[0][1] + ell[1][1] + tol ]
        ell_test_tr = [ell[0][0] + ell[1][0] + tol , ell[0][1] + ell[1][1] + tol ]
        ell_test_bl = [ell[0][0] - ell[1][0] - tol , ell[0][1] - ell[1][1] - tol ]
        ell_test_br = [ell[0][0] + ell[1][0] + tol , ell[0][1] - ell[1][1] - tol ]

        # test if the center of the found ellipses is within the bounding box of the test ellipse
        overlaps1 = ((ell[0][0] <= test_tr[0]) and (ell[0][0] >= test_tl[0]) 
                    and (ell[0][1] <= test_tl[1]) and (ell[0][1] >= test_bl[1])) 
        
        if overlaps1:
            return True
        
        # also want to test whether the center of the test ellipse is within the bounding box of the ellipses
        overlaps2 = ((ellipse[0][0] <= ell_test_tr[0]) and (ellipse[0][0] >= ell_test_tl[0]) 
                    and (ellipse[0][1] <= ell_test_tl[1]) and (ellipse[0][1] >= ell_test_bl[1])) 
        
        if overlaps2:
            return True
       

        overall_overlaps = overlaps1 or overlaps2
        if overall_overlaps == True:
            return True
        
    return False

def build_ellipses(num_ellipses):
    
    ellipse_data = []
    ellipse_list = []
    ellipse_coords_list = []
    buffer = 1

    # things needed for building the ellipse. Syntax is 
    # ellipse = ((), (), _), 
    # where the first element is the location of the centerpoint, the second 
    # element is the length of the axes along x and y and the third value is the 
    # angle between the x-axis of the Cartesian base and the corresponding semi-axis
    # print(f'xmax is {xmax} and ymax is {ymax}')
    for ellipse_num in range(num_ellipses):
        x_axis = int(random.random() * min(xmax, ymax)/20 + 1)
        y_axis = int(random.random() * x_axis) + 2
        center_x = random.random() * ((xmax - x_axis - buffer) - (xmin + x_axis + buffer)) + xmin + x_axis + buffer
        center_y = random.random() * ((ymax - x_axis - buffer) - (ymin + x_axis + buffer)) + ymin + x_axis + buffer
        center = (center_x, center_y)
        angle = int(random.random() * 180)
        ellipse = [center, (x_axis, y_axis), angle]

        ell = Point(ellipse[0][0], ellipse[0][1]).buffer(1)
        ell_scaled = scale(ell, int(ellipse[1][0]), int(ellipse[1][1]))
        ell_rotated = rotate(ell_scaled, ellipse[2])
        ell_coords = list(ell_rotated.exterior.coords)
        ell_coords = [(int(c1), int(c2)) for (c1, c2) in ell_coords]
        ell_line = LineString(ell_rotated.exterior.coords)

        # insert requirements for ellipse construction
        while (x_axis < 1 or x_axis > 7 or ell_overlaps(ellipse, ellipse_data)):
        
            x_axis = int(random.random() * min(xmax, ymax)/4 + 1)
            y_axis = int(random.random() * x_axis) + 2
            center_x = random.random() * ((xmax - x_axis - buffer) - (xmin + x_axis + buffer)) + xmin + x_axis + buffer
            center_y = random.random() * ((ymax - x_axis - buffer) - (ymin + x_axis + buffer)) + ymin + x_axis + buffer
            center = (center_x, center_y)
            angle = int(random.random()) * (180)
            ellipse = [center, (x_axis, y_axis), angle]

            ell = Point(ellipse[0][0], ellipse[0][1]).buffer(1)
            ell_scaled = scale(ell, int(ellipse[1][0]), int(ellipse[1][1]))
            ell_rotated = rotate(ell_scaled, ellipse[2])
            ell_coords = list(ell_rotated.exterior.coords)
            ell_coords = [(int(c1), int(c2)) for (c1, c2) in ell_coords]
            ell_line = LineString(ell_rotated.exterior.coords)
        
        ellipse_list.append(ell_line)
        ellipse_coords_list.append(ell_coords)
        ellipse_data.append(ellipse)

    return ellipse_list, ellipse_coords_list


# def check_intersections(obstacles):

#     # the obstacles list has shape [a][b], where a is the shape type (rectangle, circle, etc.) and 
#     # b is one of the LINESTRINGS for that shape. Idea is to just loop through and check what's disjoint

#     # print(obstacles)
#     updated_obstacles = []

#     # start by assuming True
    
#     for shape_type_idx in range(len(obstacles)):


#         for shape in obstacles[shape_type_idx]:

#             disjointed = True
#             # check that the line is disjoint with the other lines in the obstacle list

#             for other_type_idx in range(len(obstacles)):

#                 if other_type_idx == shape_type_idx:
#                     continue  # because we want to look beyond the current shape type

#                 # move on to next loop
#                 for other_shape in obstacles[other_type_idx]:
#                     if (not shape.disjoint(other_shape)) or shape.within(other_shape) or other_shape.within(shape): # the shapes are not disjoint
#                         disjointed = False
#                         break # exit the for loop

                
#                 if disjointed == False:
#                     break # exit this for loop and move on to the next shape

#             if disjointed:
#                 updated_obstacles.append(shape)

#     # print(f'New obstacle list: {updated_obstacles}')
            
#     return updated_obstacles

def check_intersections(obstacles):
    """
    Given a list of shape lists (rectangles, circles, ellipses), return a filtered list
    that contains only non-overlapping shapes. Keeps the first shape encountered and
    skips any new shape that intersects or is contained within another.
    """
    updated_obstacles = []

    # Flatten the list of obstacle types into a single list of shapes
    all_shapes = [shape for shape_type_list in obstacles for shape in shape_type_list]

    for shape in all_shapes:
        has_conflict = False
        for existing_shape in updated_obstacles:
            # if (not shape.disjoint(existing_shape)) or \
            #    shape.within(existing_shape) or \
            #    existing_shape.within(shape):
            if shape.intersects(existing_shape):
                has_conflict = True
                break
        if not has_conflict:
            updated_obstacles.append(shape)

    return updated_obstacles



#### Currently the code checks for the self-intersection of a shape with another shape of its own type
#### needs to check for the intersection of shape of one type with a shape of another type

outside = LineString([bl, br, tr, tl, bl])
wall_es = [outside]

obstacles = []
obstacle_coords = []

# num_shapes = 10

rectangles, rectangle_coords = build_rectangles(10)
circles, circle_coords = build_circles(10)
ellipses, ellipse_coords = build_ellipses(10)


obstacles.append(rectangles)
obstacles.append(circles)
obstacles.append(ellipses)


print(f'Rectangle list is: {rectangles}')
print(f'Circle list is {circles}')
print(f'Ellipse list is {ellipses}')

obstacle_coords.append(rectangle_coords)
obstacle_coords.append(circle_coords)
obstacle_coords.append(ellipse_coords)

# print(f'Obstacles are: {obstacles}')
# print(f'Obstacle coords are: {obstacle_coords}')

# check for intersections
updated_obstacles = check_intersections(obstacles)
# updated_obstacles = obstacles

# for line1 in updated_obstacles:
#     for line2 in line1:
#         wall_es.append(line2)


# print(f'updated_obstacles: {updated_obstacles}')

for line in updated_obstacles:
        wall_es.append(line)

# print(f'wall_es is {wall_es}')

walls = prep(MultiLineString(wall_es))

# Visualization Utility
class Visualization:
    def __init__(self, batch_index):
        self.batch_index = batch_index
        self.fig, self.ax = plt.subplots(num=f'Batch {batch_index}')
        self.setup_axes()

    def setup_axes(self):
        # Clear the current, or create a new figure
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_xticks(xlabels)
        self.ax.set_yticks(ylabels)
        self.ax.set_aspect('equal')

        # # Show the walls
        # for l in walls.context.geoms:
        #     self.ax.plot(*l.xy, 'k', linewidth=2)
        # if bonus in walls.context.geoms:
        #     self.ax.plot(*bonus.xy, 'b:', linewidth=3)

        # Show the walls
        for l in walls.context.geoms:
            self.ax.plot(*l.xy, 'k', linewidth=2)

        # Show
        self.show()

    def show(self, text = ''):
        # Show the plot
        plt.pause(0.001)

        # If text is specified, print and wait for confirmation
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        self.ax.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        self.ax.plot((head.x, tail.x), (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

# MAIN
def main():

    batch_size = 1
    STEP_SIZE = 0.5
    NMAX = 50

    print('Running with step size ', STEP_SIZE, ' and up to ', NMAX, ' nodes.')

    # Create the figure
    visuals = [Visualization(b) for b in range(batch_size)]

    # Visualize the start and goal nodes
    # for visual in visuals:
    #     visual.drawNode(startnode, color='orange', marker='o')
    #     visual.drawNode(goalnode, color='purple', marker='o')
    #     visual.show('Showing basic world') 

    ans = int(input('Enter 0 to end.'))
    
if __name__ == "__main__":
    main()
    # cProfile.run('main()')