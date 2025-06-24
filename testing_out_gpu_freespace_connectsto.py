from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
import torch

(xmin, xmax) = (0, 30)
(ymin, ymax) = (0, 20)

(xA, xB, xC, xD, xE) = ( 5, 12, 15, 18, 21)
(yA, yB, yC, yD)     = ( 5, 10, 12, 15)

x1 = 0
y1 = 10
x2 = 15
y2 = 5

wall1   = LineString([[xmin, yB], [xC, yB]])
wall2   = LineString([[xD, yB], [xmax, yB]])
wall3   = LineString([[xB, yC], [xC, yC]])
wall4   = LineString([[xC, yC], [xC, ymax]])
wall5   = LineString([[xC, yB],[xC, yA]])
wall6   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])

outside = LineString([[xmin, ymin], [xmax, ymin], [xmax, ymax],
                      [xmin, ymax], [xmin, ymin]])

walls = prep(MultiLineString([outside, wall1, wall2, wall3, wall4, wall5, wall6, bonus]))

device='cuda'


# IT PASSED MY TRIALS LOL
def inFreespace(next_node):

    # Returns False if any of the conditions fails
    in_bounds_mask = ((next_node[:,0] >= xmin) & (next_node[:,0] <= xmax)) | ((next_node[:,1] >= ymin) & (next_node[:,0] <= ymax))

    # Everything in the mask evaluated to False. No survivors
    if not in_bounds_mask.any():
        return in_bounds_mask

    # In the initializing, we assume that all the nodes are disjoint. the for loop tries to prove this wrong
    all_walls_disjoint_mask = torch.ones(len(next_node), dtype=bool, device='cuda')

    # check for intersection with the walls
    wall_list = [wall1, wall2, wall3, wall4, wall5, wall6, bonus]
    for wall in wall_list:
        wall_coords = wall.coords
        x1, y1, x2, y2 = wall_coords[0][0], wall_coords[0][1], wall_coords[1][0], wall_coords[1][1]

        # first, check if the wall is vertical or not

        # if the wall is vertical
        if (x2-x1 == 0):
            wall_intercept = 0
            # if the wall is vertical, then the point and wall intersect if the point has the same x
            # value and is between ymin and ymax of the wall


            # in other words, the point and line are disjoint if they have different x values or if the y of the
            # point is out of range of the line

            # check the two cases where they are disjoint

            # case 1: x-values are not equal
            is_disjoint_mask1 = x1 != next_node[:,0]

            # case 2: we-don't care about equality but the y-values are outside of the range
            is_disjoint_mask2 = next_node[:,0] > max(y1, y2)
            is_disjoint_mask3 = next_node[:,0] < min(y1, y2)

            # so is_disjoint_mask = case 1 or case 2
            is_disjoint_mask = is_disjoint_mask1 | (is_disjoint_mask2 & is_disjoint_mask3)

        # the wall is not vertical
        else:
            wall_grad = (y2-y1)/(x2-x1)
            wall_intercept = y1 - wall_grad * x1

            # Returns true if they are disjoint and returns false if they intersect
            is_disjoint_mask = (wall_grad * next_node[:,0] + wall_intercept) != (next_node[:,1])

        all_walls_disjoint_mask = all_walls_disjoint_mask & is_disjoint_mask

    return all_walls_disjoint_mask & is_disjoint_mask & in_bounds_mask





def connectsTo(nearnode, nextnode):
    
    # Create a massive list of all the x3, y3, x4, y4 values
    x3, y3 = nearnode[:,0], nearnode[:,1]
    x4, y4 = nextnode[:,0], nextnode[:,1]

    # need to make a vertical line mask for the data points
    is_vertical = x3 == x4
    is_horizontal = y3 == y4


    all_connect_check = torch.ones(len(nearnode), dtype=bool, device='cuda')

    wall_list = [wall1, wall2, wall3, wall4, wall5, wall6, bonus]
    for wall in wall_list:
        wall_coords = wall.coords
        x1, y1, x2, y2 = wall_coords[0][0], wall_coords[0][1], wall_coords[1][0], wall_coords[1][1]

        # case 1: Wall is not a vertical line
        if (x1 != x2):

            # define the wall line parameters
            wall_grad = (y2-y1)/(x2-x1)
            wall_intercept = y1 - wall_grad * x1

            # define the test-line parameters (which depends on whether it's vertical)
            line_grad = ((y4 - y3)/(x4 - x3))
            line_intercept = y4 - line_grad * x4

            # also need to account for the case where both lines are horizontal and parallel (not collinear)

            # create a mask to check that the y-values for the test_line are 'equal' and the y_wall != y_line
            parallel_horizontal_mask = torch.abs(y1 - y3) > 1e-4
            parallel_lines_mask = is_horizontal & parallel_horizontal_mask

            collinear_horizontal_mask = torch.abs(y1 - y3) <= 1e-4
            collinear_lines_mask = is_horizontal & collinear_horizontal_mask

            x_intersect = (line_intercept - wall_intercept)/(wall_grad - line_grad)
            x_intersect[is_vertical] = x4[is_vertical]

            y_intersect_wall = wall_grad * x_intersect + wall_intercept
            y_intersect_line = line_grad * x_intersect + line_intercept

            within_range = (y_intersect_wall >= torch.minimum(y3, y4)) & (y_intersect_wall <= torch.maximum(y3, y4))

            # True means that x_intersect is not on the line and False means x_intersect is on the line
            # x_intersect = not(x_intersect is on the test line)
            x_intersect_check_line = ~((x_intersect >= torch.minimum(x3, x4)) & (x_intersect <= torch.max(x3, x4)))

            # True means the y-values are the same. False means the y-values are different
            y_intersect_check = torch.abs(y_intersect_wall - y_intersect_line) <= 1e-4

            # For case_connects to be true, x_intersect must be true and y_intersect must be true
            case_connects = x_intersect_check_line & y_intersect_check
            
            case_connects = case_connects & ~within_range

            case_connects[parallel_lines_mask] = True

            case_connects[collinear_lines_mask] = False
            

        # case 2: Wall is a vertical line x = 2, for example
        elif (x1 == x2):

            line_grad = ((y4 - y3)/(x4 - x3))
            line_intercept = y4 - line_grad * x4

            x_intersect = x1 # if the wall is vertical, x2 = x1 and this is the only place the test line could intersect
            y_intersect_line = line_grad * x_intersect + line_intercept

            # there would be a connection if y_intersect is outside the max and min y-values of the wall

            within_rangex = (y_intersect_line >= min(y1, y2)) & (y_intersect_line <= max(y1, y2))
            within_rangey = (x_intersect >= torch.minimum(x3, x4)) & (x_intersect <= torch.maximum(x3, x4))
            case_connects = ~ (within_rangex & within_rangey)

            # do a separate check for if the test line is vertical too
            if torch.any((x1 == x3[is_vertical])):
                other_check = torch.abs(x3 - x1) <= 1e-4
                lines_equal_mask = is_vertical & other_check
                case_connects[lines_equal_mask] = False


        all_connect_check = all_connect_check & case_connects

    return all_connect_check


# # let's start by testing out Freespace

# # let's define a tensor with the following points:

# test_nodes = torch.tensor([[6,8], [12,6], [15, 7], [13.5, 2.5], [18, 10], [20, 11], [32, 22], [23, 10], [5,5]], device=device)
# # From the freeSpace test, this should return [False, False, False, False, False, True, False, False, True]

# # the last entry should be False because it's outside of the bounds of the

# free = inFreespace(test_nodes)
# print(test_nodes)
# print(free)


# let's check the connectsTo function




# def connectsTo2(nearnode, nextnode):
    
#     # Create a massive list of all the x3, y3, x4, y4 values
#     x3, y3 = nearnode[:,0], nearnode[:,1]
#     x4, y4 = nextnode[:,0], nextnode[:,1]

#     tolerance = 1e-4

#     # need to make a vertical line mask for the data points
#     is_vertical = torch.isclose(x3, x4, atol=tolerance)
#     is_horizontal = torch.isclose(y3, y4, atol=tolerance)


#     all_connect_check = torch.ones(len(nearnode), dtype=bool, device='cuda')

#     wall_list = [wall1, wall2, wall3, wall4, wall5, wall6, bonus]

#     for wall in wall_list:
#         wall_coords = wall.coords
#         x1, y1, x2, y2 = wall_coords[0][0], wall_coords[0][1], wall_coords[1][0], wall_coords[1][1]

#         x1_tensor = torch.tensor(x1, device=device)
#         x2_tensor = torch.tensor(x2, device=device)
#         y1_tensor = torch.tensor(y1, device=device)
#         y2_tensor = torch.tensor(y2, device=device)

#         # case 1: Wall is not a vertical line
#         if abs(x1 - x2) > tolerance:

#             # define the wall line parameters
#             wall_grad = (y2-y1)/(x2-x1)
#             wall_intercept = y1 - wall_grad * x1
            
#             # define the test-line parameters (which depends on whether it's vertical)

#             # redefine the line_grad variable to better account for the case where the line is vertical
#             # it basically says that if dx=0, then we assign infinity as the gradient. Otherwise, we do the usual
#             # gradient calculation
#             dx = x4 - x3
#             line_grad = torch.where(dx.abs() < tolerance, torch.tensor(float('inf'), device=device), (y4 - y3) / dx)

            
#             line_intercept = y4 - line_grad * x4

#             # similar to the case for the gradient, we also 'hard-code' the values for x-intersect in the case where
#             # the test line is horizontal. Otherwise, we calculate as normal
#             x_intersect = torch.where(is_vertical,x4,(line_intercept - wall_intercept) / (wall_grad - line_grad))
#             y_intersect_wall = wall_grad * x_intersect + wall_intercept

#             # need to come up with another unique case because currently, when dealing with a vertical line
#             # where y_intersect_line = nan, line_grad = inf, line_intercept = nan it says that the lines don't intersect
#             # however, the lines may intersect. How do we know this? If we calculate y_intersect_wall, and this y-value is
#             # in the range of y-values of the vertical line

#             unique_case_mask = line_grad == float('inf')

#             y_intersect_line = line_grad * x_intersect + line_intercept

#             # within_range = (y_intersect_wall >= torch.minimum(y3, y4)) & (y_intersect_wall <= torch.maximum(y3, y4))

#             # we need to now define a new intersection condition based on 
#             # 1. whether the y-intersect values of the wall and line are equal (within some small tolerance)
#             # 2. whether the y-intersect value falls between the y-range of the wall
#             # 3. whether the c-intersect value falls within the x-domain of the test lines


#             within_rangex = (y_intersect_wall >= min(y1, y2) - tolerance) & (y_intersect_wall <= max(y1, y2) + tolerance)
#             within_rangey = (x_intersect >= torch.minimum(x3, x4) - tolerance) & (x_intersect <= torch.maximum(x3, x4) + tolerance)

#             y_close = torch.abs(y_intersect_line - y_intersect_wall) <= tolerance

#             intersects = y_close & within_rangex & within_rangey

#             unique_rangey = (y_intersect_wall >= torch.minimum(y3, y4) - tolerance) & (y_intersect_wall <= torch.maximum(y3, y4) + tolerance)
#             unique_rangex = x_intersect == x3
#             intersects[unique_case_mask] = unique_rangey[unique_case_mask] & unique_rangex[unique_case_mask]

#             # check if any wall and line pair is parallel
#             # to be parallel, the near/next node pair must be horizontal and y1 and y2 must have the 'same' y-value
#             parallel_mask = is_horizontal & torch.isclose(torch.tensor(y1, device=device), y3, atol=tolerance)
            
#             # check if any wall and line pair is collinear
#             # to be collinear, the lines must first pass the parallel line test and must also overlap in x
#             collinear_mask = parallel_mask & ((torch.maximum(torch.minimum(x3, x4), torch.minimum(x1_tensor, x2_tensor)) <= torch.minimum(torch.maximum(x3, x4), torch.maximum(x1_tensor, x2_tensor)) + tolerance))

#             # Block connection if it intersects
#             # first copies over the intersects mask to a new mask called block
#             block = intersects.clone()

#             # of the members of the block mask, assign true to the collinear entried
#             block[collinear_mask] = True
            
#             # if not collinear, then assign false when adding to all_connect_check
#             all_connect_check = all_connect_check & ~block

#         # case 2: Wall is a vertical line x = 2, for example
#         elif (x1 == x2):

#             line_grad = ((y4 - y3)/(x4 - x3))
#             line_intercept = y4 - line_grad * x4

#             x_intersect = x1 # if the wall is vertical, x2 = x1 and this is the only place the test line could intersect
#             y_intersect_line = line_grad * x_intersect + line_intercept

#             # there would be a connection if y_intersect is outside the max and min y-values of the wall

#             within_rangex = (y_intersect_line >= min(y1, y2) - tolerance) & (y_intersect_line <= max(y1, y2) + tolerance)
#             within_rangey = (x_intersect >= torch.minimum(x3, x4) - tolerance) & (x_intersect <= torch.maximum(x3, x4) + tolerance)
            
#             intersects = within_rangex & within_rangey

#             # A special case where the wall and line are both vertical and overlap
            
#             # a mask that checks whether the vertical wall and vertical wall have the same x-value
#             vertical_overlap = is_vertical & (torch.abs(x3 - x1) <= tolerance)

            
#             # checks whether two lines overlap
#             overlap_y_range = ((torch.maximum(torch.minimum(y3, y4), torch.minimum(y1_tensor, y2_tensor)) <= torch.minimum(torch.maximum(y3, y4), torch.maximum(y1_tensor, y2_tensor)) + tolerance))
           
           
#             # define another mask that checks for collinear lines (both are vertical, with the same x-value and occupy some of the same y-values)
#             collinear_vertical_block = vertical_overlap & overlap_y_range
            
#             # this mask checks whether the line/wall pair intersects or is collinear
#             block = intersects | collinear_vertical_block

#             # if the line/wall pair intersects or is collinear then we say the nearnode and nextnode don't connect (connect = False) because
#             # they are blocked by an intersection with the wall
#             all_connect_check = all_connect_check & ~block

#     return all_connect_check




def connectsTo2(nearnode, nextnode):
    """
    Returns a boolean tensor indicating which node pairs can connect without intersecting walls.
    
    Inputs:
    - nearnode: (B, 2) tensor of start points
    - nextnode: (B, 2) tensor of end points
    
    Output:
    - (B,) boolean tensor: True if the segment is not blocked, False if it intersects any wall
    """

    
    B = nearnode.shape[0]
    
    def orientation(p, q, r):
        """Return orientation of triplet (p, q, r) as 0, 1, or 2"""
        # 0 --> colinear, 1 --> clockwise turn, 2 --> counterclockwise turn
        # val performs the cross product to get the orientation
        # val == 0 --> colinear, val > 0 --> clockwise, val < 0 --> counterclockwise
        val = (q[:,1] - p[:,1]) * (r[:,0] - q[:,0]) - (q[:,0] - p[:,0]) * (r[:,1] - q[:,1])
        zero = torch.tensor(0.0, device=val.device)
        return torch.where(
            torch.abs(val) < 1e-8,
            torch.tensor(0, device=val.device),  # colinear
            torch.where(val > 0, torch.tensor(1, device=val.device), torch.tensor(2, device=val.device))  # clockwise/counterclockwise
        )

    def on_segment(p, q, r):
        """Check if point q lies on segment pr"""
        return (
            (q[:,0] <= torch.max(p[:,0], r[:,0])) & (q[:,0] >= torch.min(p[:,0], r[:,0])) &
            (q[:,1] <= torch.max(p[:,1], r[:,1])) & (q[:,1] >= torch.min(p[:,1], r[:,1]))
        )

    def segment_intersects(p1, p2, q1, q2):
        """Returns a (B,) boolean tensor if segment p1-p2 intersects q1-q2"""
        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        # a general intersection test
        general = (o1 != o2) & (o3 != o4)

        # check if the lines are colinear and overlap/touch
        col1 = (o1 == 0) & on_segment(p1, q1, p2)
        col2 = (o2 == 0) & on_segment(p1, q2, p2)
        col3 = (o3 == 0) & on_segment(q1, p1, q2)
        col4 = (o4 == 0) & on_segment(q1, p2, q2)

        return general | col1 | col2 | col3 | col4

    all_connect_check = torch.ones(B, dtype=torch.bool, device=device)

    wall_list = [wall1, wall2, wall3, wall4, wall5, wall6, bonus]

    for wall in wall_list:
        wx1, wy1 = wall.coords[0]
        wx2, wy2 = wall.coords[1]

        wall_start = torch.tensor([wx1, wy1], device=device).expand(B, 2)
        wall_end   = torch.tensor([wx2, wy2], device=device).expand(B, 2)

        intersects = segment_intersects(nearnode, nextnode, wall_start, wall_end)

        all_connect_check &= ~intersects  # Block if it intersects

    return all_connect_check


# define the nearnodes I expect this to return False, True, False, True, True, False,     True, False, False, True, False
nearnode = torch.tensor([[0,0], [1,2], [5,2], [0,0], [2,0], [15,6],   [13,8], [12,8], [0,10], [18,5], [25,5]], dtype=torch.float, device=device)
nextnode = torch.tensor([[0,10], [5,5], [15,4], [0,5], [2,2], [15,8],    [13,4], [16,8], [5,10], [22,5], [25,15]], dtype=torch.float, device=device)

ans = connectsTo2(nearnode, nextnode)
print(ans)