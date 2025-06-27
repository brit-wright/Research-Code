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

print(x1, x2, y1, y2)
print(abs(x1 - x2))
print(abs(y1 - y2))


xmin = min(x1, x2)
xmax = max(x1, x2)
ymin = min(y1, y2)
ymax = max(y1, y2)


(xA, xB, xC) = ( 5, 12, 15)
(yA, yB, yC)     = ( 5, 10, 12)

xlabels = (xmin, xA, xB, xC, xmax)
ylabels = (ymin, yA, yB, yC, ymax)

tl = [xmin, ymax]
tr = [xmax, ymax]
bl = [xmin, ymin]
br = [xmax, ymin]

print(tl, tr, bl, br)

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
    rbr_list = []
    rbl_list = []
    rtl_list = []
    rtr_list = []
    for rect_num in range(num_rectangles):

        rx1 = int(random.random() * (xmax - xmin) + xmin)
        rx2 = int(random.random() * (xmax - xmin) + xmin)
        ry1 = int(random.random() * (ymax - ymin) + ymin)
        ry2 = int(random.random() * (ymax - ymin) + ymin)

        # print(x1, x2, y1, y2)
        # print(abs(x1 - x2))
        # print(abs(y1 - y2))
        print(xmax, ymax)
        while ((abs(rx1 - rx2) <= 2 or abs(ry1 - ry2) <= 2) or (abs(rx1 - rx2) > 12 or abs(ry1 - ry2) > 10) 
            or (min(rx1, rx2) < xmin) or (max(rx1, rx2) > xmax) or (min(ry1, ry2) < ymin) or (max(ry1, ry2) > ymax) or r_overlaps(rx1, rx2, ry1, ry2, rtr_list, rtl_list, rbr_list, rbl_list)):
            rx1 = int(random.random() * (xmax - xmin) + xmin)
            rx2 = int(random.random() * (xmax - xmin) + xmin)
            ry1 = int(random.random() * (ymax - ymin) + ymin)
            ry2 = int(random.random() * (ymax - ymin) + ymin)

        print(rx1, rx2, ry1, ry2)
        print(abs(rx1 - rx2))
        print(abs(ry1 - ry2))

        rxmin = min(rx1, rx2)
        rxmax = max(rx1, rx2)
        rymin = min(ry1, ry2)
        rymax = max(ry1, ry2)

        rtl = [rxmin, rymax]
        rtr = [rxmax, rymax]
        rbl = [rxmin, rymin]
        rbr = [rxmax, rymin]

        print(rtl, rtr, rbl, rbr)

        rectangle = LineString([rbl, rbr, rtr, rtl, rbl])

        rbl_list.append(rbl)
        rbr_list.append(rbr)
        rtr_list.append(rtr)
        rtl_list.append(rtl)

        rectangle_list.append(rectangle)

    return rectangle_list

def circle_rectangle_overlap(center, radius, rbl, rbr, rtr, rtl, buffer=1.0):
    cx, cy = center
    rect_xmin = min(rbl[0], rtl[0]) - buffer
    rect_xmax = max(rbr[0], rtr[0]) + buffer
    rect_ymin = min(rbl[1], rbr[1]) - buffer
    rect_ymax = max(rtl[1], rtr[1]) + buffer

    closest_x = max(rect_xmin, min(cx, rect_xmax))
    closest_y = max(rect_ymin, min(cy, rect_ymax))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy < (radius + buffer)**2

def cr_overlaps(center, radius, rbl_list, rbr_list, rtr_list, rtl_list):
    for i in range(len(rbl_list)):
        if circle_rectangle_overlap(center, radius, rbl_list[i], rbr_list[i], rtr_list[i], rtl_list[i]):
            return True
    return False

def c_overlaps(test_data, circle_data):
    overlaps = False

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
    print(f'xmax is {xmax} and ymax is {ymax}')
    for ellipse_num in range(num_ellipses):
        x_axis = int(random.random() * min(xmax, ymax)/4 + 1)
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
        ellipse_coords_list.extend(ell_coords)
        ellipse_data.append(ellipse)

    return ellipse_list

outside = LineString([bl, br, tr, tl, bl])
wall_es = [outside]
# rectangles = build_rectangles(5)
# circles = build_circles(5)
# for r in rectangles:
#     wall_es.append(r)
# for c in circles:
#     wall_es.append(c)
ellipses = build_ellipses(10)
for el in ellipses:
    wall_es.append(el)
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