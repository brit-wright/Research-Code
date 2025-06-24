### REWRITING inFreespace and connectsTo

#### Let's start by rewriting inFreespace

### Here's the original code for inFreespace

# def inFreespace(nextnode_cpu_list):
#     # nextnode_cpu_list is a list of size batch_size which contains the (x, y) tuple
#     for node_cpu in nextnode_cpu_list:
#         x, y = node_cpu
#         freespace = []
#         if (x <= xmin or x >= xmax or y <= ymin or y >= ymax):
#             ans = False
#             freespace.append(False)
#         else:
#             ans = walls.disjoint(Point(x,y))
#             freespace.append(ans)
#     freespace_list = np.array(freespace)
#     return freespace_list

# The main thing to be rewritten is disjoint.

# We can start with a point and line example

# first we define the line 

from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep

(xmin, xmax) = (0, 30)
(ymin, ymax) = (0, 20)

(xA, xB, xC, xD, xE) = ( 5, 12, 15, 18, 21)
(yA, yB, yC, yD)     = ( 5, 10, 12, 15)

xlabels = (xmin, xA, xB, xC, xD, xE, xmax)
ylabels = (ymin, yA, yB, yC, yD,     ymax)

# Draw the outer boundary/walls
outside = LineString([[xmin, ymin], [xmax, ymin], [xmax, ymax],
                      [xmin, ymax], [xmin, ymin]])

# Draw the interior walls that the mattress will have to move around

x1 = 0
y1 = 10
x2 = 15
y2 = 5

wall1   = LineString([[x1, y1], [x2, y2]])
wall2   = LineString([[xD, yB], [xmax, yB]])
wall3   = LineString([[xB, yC], [xC, yC], [xC, ymax]])
wall4   = LineString([[xC, yB],[xC, yA]])
wall5   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])

walls = prep(MultiLineString([outside, wall1, wall2, wall3, wall4, wall5, bonus]))

# wall1 = (0, 10), (15, 5)

# True means A and B do not share any point in space
# False means that they overlap

# This is how I plan to do it

# 1: Calculate the gradient of the line
# 2. Calculate the y-intercept of the line
# 3. Now that we have the equation of the line we can say 
# that if gradient *x + y-intercept is equal to the y value of
# the point then there is overlap

# Testing for point-to-line disjointness

wall_grad = (10-5)/(0-15)  # (y2-y1)/(x2-1)
wall_y_intercept = 10 - wall_grad * 0 # y2 - wall_grad * x2

test_point1 = (6, 8)
test_point2 = (6, 10)

# # Expect to get False
# print(wall1.disjoint(Point(6,8)))
# print(wall_grad * test_point1[0] + wall_y_intercept != test_point1[1])

# # Expect to get True
# print(wall1.disjoint(Point(6,10)))
# print(wall_grad * test_point2[0] + wall_y_intercept != test_point2[1])

# Testing for line-to-line disjointness

# define second line
x3 = 1
y3 = 2
x4 = 6
y4 = 12

# Basically we wanna check if the lines intersect
# We do this by making both of these into the equation of a line
# 

# Using our example of wall 1 again:

# First create the equation of the line for the wall:

line = LineString([(x3, y3),(x4, y4)])


wall_grad = (y1-y2)/(x1-x2)  # (y2-y1)/(x2-1)
wall_y_intercept = y1 - wall_grad * x1 # y2 - wall_grad * x2

test_line_grad = (y3-y4)/(x3-x4)
test_line_intercept = y4 - test_line_grad * x4

# to check if the lines intersect, first confirm that the gradients
# are not equal and the line intercepts are not equal.

# disjoint returns true if the lines do not intersect and false if
# they do intersect

# Expect to get False (the lines do intersect)
print(wall1.disjoint(line))

# print(wall_grad == test_line_grad)
# this one doesn't work because it doesn't account for the endpoints
# it assumes the lines go on until infinity
# we also need to account for the actual location where both lines intersect

# Line 1: y1 = m1 * x1 + c1
# Line 2: y2 = m2 * x2 + c2

# After confirming that the gradients are different we can do the following
# 1. equate m1 * x + c1 = m2 * x + c2 to get the x-location of the point of intersection
# m1 * x + c1 = m2 * x + c2 --> x = (c2 - c1)/(m1 - m2)
# 2. Then substitute this x-value into the equation to get the y-location of the point of
# intersection
# 3. Then change that the x and y location of the point of intersection are actually between
# the x and y values of the two lines

if (wall_grad == test_line_grad) and (wall_y_intercept == test_line_intercept):
    print(False)

elif (wall_grad != test_line_grad):
    # confirmed that the gradients are not equal
    # then we calculate the x-value for the point of intersection
    x_intersect = (test_line_intercept - wall_y_intercept)/(wall_grad - test_line_grad)

    # then find the corresponding y-location
    y_intersect = wall_grad*x_intersect + wall_y_intercept
    y_intersect_line = test_line_grad*x_intersect+test_line_intercept

    # then we should do the checks to ensure the point of intersection is within the correct
    # bounds
    if y_intersect_line == y_intersect:
        print('False') # not disjoint
    else:
        print('True') # they are disjoint
    # # Returns False if the two lines intersect (we expect this result)
    # if x_intersect > min(x3, x1) and x_intersect < min(x2, x4) and y_intersect < min(y1, y4) and y_intersect > min (y2, y3):
    #     print(False)
    # else:
    #     print(True)
else:
    print(True)


########### STORING THESE HERE FOR NOW:

"""
def inFreespace(next_node):

    # Returns False if any of the conditions fails
    in_bounds_mask = next_node[:,0] >= xmin or next_node[:,0] <= xmax or next_node[:,1] >= ymin or next_node[:,0] <= ymax

    # Everything in the mask evaluated to False. No survivors
    if not in_bounds_mask.any():
        return in_bounds_mask

    # In the initializing, we assume that all the nodes are disjoint. the for loop tries to prove this wrong
    all_walls_disjoint_mask = torch.ones(len(next_node), dtype=bool, device='cuda')

    # check for intersection with the walls
    for wall in walls:
        wall_coords = wall.coords
        x1, y1, x2, y2 = wall_coords[0][0], wall_coords[0][1], wall_coords[1][0], wall_coords[1][1]
        wall_grad = (y2-y1)/(x2-x1)
        wall_intercept = y1 - wall_grad * x1

        # Returns true if they are disjoint and returns false if they intersect
        is_disjoint_mask = wall_grad * next_node[:,0] + wall_intercept != next_node[:,1]

        all_walls_disjoint_mask = all_walls_disjoint_mask & is_disjoint_mask

    return all_walls_disjoint_mask & in_bounds_mask


def connectsTo(nearnode_cpu_list, nextnode_cpu_list):
    connects = []
    for i in range(0,len(nearnode_cpu_list)):
        x1, y1 = nearnode_cpu_list[i]
        x2, y2 = nextnode_cpu_list[i]

        line = LineString([(x1, y1), (x2, y2)])

        ans = walls.disjoint(line)
        connects.append(ans)
    connects_list = np.array(connects)
    return connects_list
"""


"""
Storing this here for now:

# check if the lines are collinear
        # returns True if collinear and False otherwise
        collinear_mask = wall_grad == line_grad & wall_intercept == line_intercept

        # check if the lines are parallel
        # returns True if parallel and False otherwise
        parallel_mask = wall_grad == line_grad & wall_intercept != line_intercept
"""