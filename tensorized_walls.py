import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
from vandercorput       import vandercorput
import cProfile
import torch
# seed = int(random.random()*10000)
# seed = 9676
seed = 3331
random.seed(seed)
# print(f"{seed=}")


STEP_SIZE = 0.25
# NMAX = 90   # Set the maximum number of nodes
# SMAX = 90   # Set the maximum number of steps

NMAX = 10000
SMAX = 10000


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
wall1   = LineString([[xmin, yB], [xC, yB]])
wall2   = LineString([[xD, yB], [xmax, yB]])
wall3   = LineString([[xB, yC], [xC, yC]])
wall4   = LineString([[xC, yC], [xC, ymax]])
wall5   = LineString([[xC, yB],[xC, yA]])
wall6   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])

# Let's start by converting the walls shapely things to actual tensors.
# Tensors should be of shape (num_walls, [[start coords], [end coords]])

device='cuda'
twall1   = [[xmin, yB], [xC, yB]]
twall2   = [[xD, yB], [xmax, yB]]
twall3   = [[xB, yC], [xC, yC]]
twall4   = [[xC, yC], [xC, ymax]]
twall5   = [[xC, yB],[xC, yA]]
twall6   = [[xC, ymin], [xB, yA]]
tbonus   = [[xD, yC], [xE, yC]]
twalls = [twall1, twall2, twall3, twall4, twall5, twall6, tbonus]
wall_coords = torch.tensor([[[twall[0][0],twall[0][1]], [twall[1][0],twall[1][1]]] for twall in twalls], dtype=torch.float, device=device)
W = wall_coords.shape[0]
# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
walls = prep(MultiLineString([outside, wall1, wall2, wall3, wall4, wall5, wall6, bonus]))
# walls = prep(MultiLineString([outside, wall1, wall2, wall3]))

# Define the start/goal states (x, y, theta) of the mattress
(xstart, ystart) = (xA, yD)
(xgoal, ygoal) = (5, 5)

    
# Converting the vectorized (orientation-version of) Freespace function into tensor logic
def inFreespace2(next_node):

    # Returns False if any of the conditions fails
    in_bounds_mask = ((next_node[:,0] >= xmin) & (next_node[:,0] <= xmax)) & ((next_node[:,1] >= ymin) & (next_node[:,1] <= ymax))
    print(in_bounds_mask)
    print(in_bounds_mask.shape[0])
    B = next_node.shape[0]
    
    p2 = next_node.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nextnode to (B, W, 2)
    print(p2)
    # also want to change the wall coords to have the same (B, W, 2) shape
    # the shape of wall_coords is, by default, (W, 2, 2)

    # We separate the walls into their end-points q1 and q1. We then want to unsqueeze and expand each in order to 
    # get the (B, W, 2) shape. Note that unsqueeze(0) gives the shape [1, 7, 2]. Expansing (B, -1, -1) means that we 
    # wanna change the first dimension to B and leave the other dimensions as they are

    q1 = wall_coords[:,0,:].unsqueeze(0).expand(B, -1, -1)
    q2 = wall_coords[:,1,:].unsqueeze(0).expand(B, -1, -1)
    
    def orientation(p, q, r):
        """Return orientation of triplet (p, q, r) as 0, 1, or 2"""
        # 0 --> colinear, 1 --> clockwise turn, 2 --> counterclockwise turn
        # val performs the cross product to get the orientation
        # val == 0 --> colinear, val > 0 --> clockwise, val < 0 --> counterclockwise
        val = (q[:,:,1] - p[:,:,1]) * (r[:,:,0] - q[:,:,0]) - (q[:,:,0] - p[:,:,0]) * (r[:,:,1] - q[:,:,1])
        return torch.where(
            torch.abs(val) < 1e-8,
            torch.tensor(0, device=val.device),  # colinear
            torch.where(val > 0, torch.tensor(1, device=val.device), torch.tensor(2, device=val.device))  # clockwise/counterclockwise
        )
    
    
    def on_segment(p, q, r):
        """Check if point q lies on segment pr"""
        return (
            (q[:,:,0] <= torch.max(p[:,:,0], r[:,:,0])) & (q[:,:,0] >= torch.min(p[:,:,0], r[:,:,0])) &
            (q[:,:,1] <= torch.max(p[:,:,1], r[:,:,1])) & (q[:,:,1] >= torch.min(p[:,:,1], r[:,:,1]))
        )

    print(q1.shape)
    print(q2.shape)
    print(p2.shape)

    o1 = orientation(q1, q2, p2)
    print(o1)
    print(o1.shape)
    # a general intersection test
    # (o1 == 0) means there is colinearity and on_segment means there's overlap. --> True means not in Freespace
    intersects = (o1 == 0) & on_segment(q1, p2, q2)
    print(intersects)
    print(intersects.shape)
    wall_intersect = intersects.any(dim=1)
    return ~wall_intersect & in_bounds_mask


# def connectsTo(nearnode, nextnode):
#     """
#     Returns a boolean tensor indicating which node pairs can connect without intersecting walls.
    
#     Inputs:
#     - nearnode: (B, 2) tensor of start points
#     - nextnode: (B, 2) tensor of end points
    
#     Output:
#     - (B,) boolean tensor: True if the segment is not blocked, False if it intersects any wall
#     """
#     B = nearnode.shape[0]

#     # Let's go back to our everything True assumption

#     # Okay this is the weird part. Basically need to do some re-sizing to prep for broadcasting 
#     # so that dimensions can become compatible.

#     # Overall, we want to expand to the shape (B, W, 2) in order to compare the Batches to the Walls

#     p1 = nearnode.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nearnode to (B, W, 2)
#     p2 = nextnode.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nextnode to (B, W, 2)

#     # also want to change the wall coords to have the same (B, W, 2) shape
#     # the shape of wall_coords is, by default, (W, 2, 2)

#     # We separate the walls into their end-points q1 and q1. We then want to unsqueeze and expand each in order to 
#     # get the (B, W, 2) shape. Note that unsqueeze(0) gives the shape [1, 7, 2]. Expansing (B, -1, -1) means that we 
#     # wanna change the first dimension to B and leave the other dimensions as they are

#     q1 = wall_coords[:,0,:].unsqueeze(0).expand(B, -1, -1)
#     q2 = wall_coords[:,1,:].unsqueeze(0).expand(B, -1, -1)

#     # We now go into vectorizing the orientation operation. Each parameter passed into orientation should have shape
#     # (B, W, 2)
#     def orientation(p, q, r):
#         """Return orientation of triplet (p, q, r) as 0, 1, or 2"""
#         # 0 --> colinear, 1 --> clockwise turn, 2 --> counterclockwise turn
#         # val performs the cross product to get the orientation
#         # val == 0 --> colinear, val > 0 --> clockwise, val < 0 --> counterclockwise
#         val = (q[:,:,1] - p[:,:,1]) * (r[:,:,0] - q[:,:,0]) - (q[:,:,0] - p[:,:,0]) * (r[:,:,1] - q[:,:,1])
        
#         return torch.where(
#             torch.abs(val) < 1e-8,
#             torch.tensor(0, device=device),  # colinear
#             torch.where(val > 0, torch.tensor(1, device=device), torch.tensor(2, device=device))  # clockwise/counterclockwise
#         )
    
#     """Returns a (B,) boolean tensor if segment p1-p2 intersects q1-q2"""
#     o1 = orientation(p1, p2, q1)
#     o2 = orientation(p1, p2, q2)
#     o3 = orientation(q1, q2, p1)
#     o4 = orientation(q1, q2, p2)
#     # a general intersection test
#     intersect = (o1 != o2) & (o3 != o4) # shape is (B, W). True if line intersects with wall
#     intersects_any_wall = intersect.any(dim=1)

#     return ~intersects_any_wall 


# # define the nearnodes I expect this to return False, True, False, True, True, False, True, False, False, True, False
# nearnode = torch.tensor([[0,0], [1,2], [5,2], [0,0], [2,0], [15,6], [13,8], [12,8], [0,10], [18,5], [25,5]], dtype=torch.float, device=device)
# nextnode = torch.tensor([[0,10], [5,5], [15,4], [0,5], [2,2], [15,8], [13,4], [16,8], [5,10], [22,5], [25,15]], dtype=torch.float, device=device)
# # connections = connectsTo(nearnode, nextnode)
# # print(connections)

# # which nodes are failing?
# # [15,6][15,8], [0,10][5,10] --> this is fine because realistically, these nodes should be weeded out by the inFreespace function


# # From the freeSpace test, this should return [True, True, False, False, False, True, False, False, True]
# test_nodes = torch.tensor([[6,8], [12,6], [15, 7], [13.5, 2.5], [18, 10], [20, 11], [32, 22], [23, 10], [5,5]], device=device)
# print(test_nodes.shape)
# freespace_test = inFreespace2(test_nodes)
# print(freespace_test)


# Writing another version of connectsTo to reduce the number of orientation calls
def connectsTo(nearnode, nextnode):
    """
    Returns a boolean tensor indicating which node pairs can connect without intersecting walls.
    
    Inputs:
    - nearnode: (B, 2) tensor of start points
    - nextnode: (B, 2) tensor of end points
    
    Output:
    - (B,) boolean tensor: True if the segment is not blocked, False if it intersects any wall
    """
    B = nearnode.shape[0]

    # Let's go back to our everything True assumption

    # Okay this is the weird part. Basically need to do some re-sizing to prep for broadcasting 
    # so that dimensions can become compatible.

    # Overall, we want to expand to the shape (B, W, 2) in order to compare the Batches to the Walls

    p1 = nearnode.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nearnode to (B, W, 2)
    p2 = nextnode.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nextnode to (B, W, 2)

    # also want to change the wall coords to have the same (B, W, 2) shape
    # the shape of wall_coords is, by default, (W, 2, 2)

    # We separate the walls into their end-points q1 and q1. We then want to unsqueeze and expand each in order to 
    # get the (B, W, 2) shape. Note that unsqueeze(0) gives the shape [B, W, 2]. Expanding (B, -1, -1) means that we 
    # wanna change the first dimension to B and leave the other dimensions as they are

    q1 = wall_coords[:,0,:].unsqueeze(0).expand(B, -1, -1) # this contains one endpoint, has shape (B, W, 2)
    q2 = wall_coords[:,1,:].unsqueeze(0).expand(B, -1, -1) # contains the other endpoint, has shape (B, W, 2)

    # let's try the bounding box method...

    # Wall: define minimum and maximum wall coordinates
    wall_mins = wall_coords.min(dim=1).values  # should be of shape (W,2)
    wall_maxs = wall_coords.max(dim=1).values  # should be of shape (W,2)

    # Wall: expand the min and max for broadcasting
    wall_mins_e = wall_mins.unsqueeze(0).expand(B, -1, -1)  # should be of shape (B, W, 2)
    wall_maxs_e = wall_maxs.unsqueeze(0).expand(B, -1, -1)  # should be of shape (B, W, 2)

    # Nodes: define the minimum and maximum coordinates
    seg_mins = torch.minimum(p1, p2) # Shape is B, W, 2
    seg_maxs = torch.maximum(p1, p2) # Shape is B, W, 2

    # Define the bounding box mask. If any entry in disjoint mask is true, it means we know for sure
    # that this wall will not intersect (without having to call the orientation function on it)

    disjoint_mask = (
        (seg_maxs[:,:,0] < wall_mins_e[:,:,0]) |
        (seg_mins[:,:,0] > wall_maxs_e[:,:,0]) |
        (seg_maxs[:,:,1] < wall_mins_e[:,:,1]) |
        (seg_mins[:,:,1] > wall_maxs_e[:,:,1])
    )

    valid_mask = ~disjoint_mask # True means that the wall should be processed

    # We now go into vectorizing the orientation operation. Each parameter passed into orientation should have shape
    # (B, W, 2)
    def orientation(p, q, r):
        """Return orientation of triplet (p, q, r) as 0, 1, or 2"""
        # 0 --> colinear, 1 --> clockwise turn, 2 --> counterclockwise turn
        # val performs the cross product to get the orientation
        # val == 0 --> colinear, val > 0 --> clockwise, val < 0 --> counterclockwise
        val = (q[:,1] - p[:,1]) * (r[:,0] - q[:,0]) - (q[:,0] - p[:,0]) * (r[:,1] - q[:,1])
        
        return torch.where(
            torch.abs(val) < 1e-8,
            torch.tensor(0, device=device),  # colinear
            torch.where(val > 0, torch.tensor(1, device=device), torch.tensor(2, device=device))  # clockwise/counterclockwise
        )
    
    """Returns a (B,) boolean tensor if segment p1-p2 intersects q1-q2"""

    # Need to re-define p1, p2, q1, q2 with valid_mask applied
    b_idx, w_idx = torch.nonzero(valid_mask, as_tuple=True) # returns the batch and wall indices where valid_mask is True

    # Get the masked p1, p2, q1, q2 tensors
    p1_masked = p1[b_idx, w_idx]
    p2_masked = p2[b_idx, w_idx]
    q1_masked = q1[b_idx, w_idx]
    q2_masked = q2[b_idx, w_idx]


    # Apply the disjoint mask to p1, p2, p3, p4 so that irrelevant walls don't get processed

    o1 = orientation(p1_masked, p2_masked, q1_masked)
    o2 = orientation(p1_masked, p2_masked, q2_masked)
    o3 = orientation(q1_masked, q2_masked, p1_masked)
    o4 = orientation(q1_masked, q2_masked, p2_masked)

    intersects = (o1 != o2) & (o3 != o4) # shape is (b, w). True if line intersects with wall

    result = torch.ones(B, dtype=torch.bool, device=device)

    if intersects.any():    # if any entry in intersect is True
        batch_collide = b_idx[intersects] # define batch_collide as the batch_indices where there are node-wall collisions
        result[batch_collide] = False # for those batches, connectsTo is False
    return result


# define the nearnodes I expect this to return False, True, False, True, True, False*, True, False, False*, True, False
nearnode = torch.tensor([[0,0], [1,2], [5,2], [0,0], [2,0], [15,6], [13,8], [12,8], [0,10], [18,5], [25,5]], dtype=torch.float, device=device)
nextnode = torch.tensor([[0,10], [5,5], [15,4], [0,5], [2,2], [15,8], [13,4], [16,8], [5,10], [22,5], [25,15]], dtype=torch.float, device=device)
connections = connectsTo(nearnode, nextnode)
print(connections)

# which nodes are failing?
# [15,6][15,8], [0,10][5,10] --> this is fine because realistically, these nodes should be weeded out by the inFreespace function


# From the freeSpace test, this should return [True, True, False, False, False, True, False, False, True]
# test_nodes = torch.tensor([[6,8], [12,6], [15, 7], [13.5, 2.5], [18, 10], [20, 11], [32, 22], [23, 10], [5,5]], device=device)
# print(test_nodes.shape)
# freespace_test = inFreespace2(test_nodes)
# print(freespace_test)
