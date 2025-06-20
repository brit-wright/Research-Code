import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
from vandercorput       import vandercorput

import torch
# seed = int(random.random()*10000)
seed = 5414
random.seed(seed)
print(f"{seed=}")


STEP_SIZE = 0.25
NMAX = 90   # Set the maximum number of nodes
SMAX = 90   # Set the maximum number of steps

NMAX = 20000
SMAX = 20000


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
wall3   = LineString([[xB, yC], [xC, yC], [xC, ymax]])
wall4   = LineString([[xC, yB],[xC, yA]])
wall5   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])

# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
walls = prep(MultiLineString([outside, wall1, wall2, wall3, wall4, wall5, bonus]))
# walls = prep(MultiLineString([outside, wall1, wall2, wall3]))

# Define the start/goal states (x, y, theta) of the mattress
(xstart, ystart) = (xA, yD)
# (xgoal, ygoal) = (10, yD)
(xgoal, ygoal) = (xA, yA)

# Visualization Utility
class Visualization:
    def __init__(self):

        # Clear the current, or create a new figure
        plt.clf()

        # Create new axes, enable the grid, and set the axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_xticks(xlabels)
        plt.gca().set_yticks(ylabels)
        plt.gca().set_aspect('equal')

        # Show the walls
        for l in walls.context.geoms:
            plt.plot(*l.xy, 'k', linewidth=2)
        if bonus in walls.context.geoms:
            plt.plot(*bonus.xy, 'b:', linewidth=3)

        # Show
        self.show()

    def show(self, text = ''):
        # Show the plot
        plt.pause(0.001)

        # If text is specified, print and wait for confirmation
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x), (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

# NODE DEFINITION

class Node:
    def __init__(self, x, y):
        # Define a parent 
        self.parent = None

        # Define/remember the state/coordinates (x,y,theta) of the node
        self.x = x
        self.y = y
        
    # Node Utilities
    # To print the node
    def __repr__(self):
        return("<Node %5.2f,%5.2f>" % (self.x, self.y))
    
    # Compute/create an intermediate node for checking the local planner
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                self.y + alpha * (other.y - self.y))
    
    # Return a tuple of coordinates to compute Euclidean distances
    def coordinates(self):
        return(self.x, self.y)
    
    # Compute the relative distance Euclidean distance to another node
    def distance(self, other):
        return dist(self.coordinates(), other.coordinates())
    
# Freespace check function
def inFreespace(nextnode_cpu_list):
    # nextnode_cpu_list is a list of size batch_size which contains the (x, y) tuple
    for node_cpu in nextnode_cpu_list:
        x, y = node_cpu
        freespace = []
        if (x <= xmin or x >= xmax or y <= ymin or y >= ymax):
            ans = False
            freespace.append(False)
        else:
            ans = walls.disjoint(Point(x,y))
            freespace.append(ans)
    freespace_list = np.array(freespace)
    return freespace_list

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

# RRT Function

def rrt(startnode, goalnode, visual):

    # Leaving many more comments because I'm genuinely confused lol

    ################### LET'S START BY DEFINING THE IMPORTANT VARIABLES FOR THIS PROBLEM ####################

    # first we need to indicate that we're changing the processing/device from CPU to GPU (cuda)
    device = torch.device('cuda')   

    # Batch Size is basically the number of RRTs I want to have running in parallel
    batch_size = 1  

    # Now I'm defining node counts which basically tells us, how many nodes are in each tree
    # where it's one tree per batch
    node_counts = torch.ones(batch_size, dtype=torch.long, device=device)

    # Defining the start and the goal node as tensors containing the x and y positions of the
    # nodes
    start = torch.tensor([[startnode.x, startnode.y]], dtype=torch.float, device=device).repeat(batch_size, 1)
    goal = torch.tensor([[goalnode.x, goalnode.y]], dtype=torch.float, device=device).repeat(batch_size, 1)

    # Next, I'm creating the tree_parents tensor of size batch_size, NMAX
    # How it works is that for each nth node (which is defined based on the location in NMAX)
    # It stores the index of that node's parent. Currently initialized to -1 meaning that 
    # None is assigned to parents.
    tree_parents = torch.full((batch_size, NMAX), -1, dtype=torch.long, device=device)

    # Next we define the tree. For each batch, the tree will store the (x,y) positions of the 
    # nth node that has been processed. The nth node is defined based on the location of the 
    # node in NMAX

    # Tree positions is also initialized for the zeroth node as the startnode. This is equivalent
    # to saying tree = [startnode]

    tree_positions = torch.zeros((batch_size, NMAX, 2), device=device)
    tree_positions[:, 0, 0] = startnode.x
    tree_positions[:, 0, 1] = startnode.y


    ########### NOW LET'S GET TO WRITING THE addtotree FUNCTION ##################
    def addtotree(valid_batches, valid_nextnodes, nearest_indices):
        # start by assigning the parent of the new_node to be the nearnode
        tree_parents[valid_batches, node_counts[valid_batches]] = nearest_indices

        # add the new node to the tree
        tree_positions[valid_batches, node_counts[valid_batches], :] = valid_nextnodes

        # increment the node cout
        node_counts[valid_batches] += 1

    ########### OKAY LET'S GO INTO THE ACUTAL LOOP LOGIC #####################
    # Also define the number of steps that have been made, since we have a constraint on the number
    # steps that are supposed to be done for each tree
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # ALSO DEFINE THE PROBABILITY
    p = 0.3

    # DEFINE THE FLAGS/STOPPERS
    active_batches = torch.ones(batch_size, dtype=torch.bool, device=device)

    ######### ENTER THE WHILE LOOP LOL #######################
    while True:
        
        # Define the random.random() tensor
        random_tensor = torch.rand(batch_size, device=device)

        # Define the boolean meask which compares each value in the random tensor to the probability variable
        mask = random_tensor <= p


        # Define the samples tensor which does non-uniform stampling
        samples = torch.empty((batch_size, 2), device=device)
        samples[:, 0] = torch.rand(batch_size, device=device) * (xmax - xmin) + xmin
        samples[:, 1] = torch.rand(batch_size, device=device) * (ymax - ymin) + ymin
        
        # Logic to check whether mask is true or not

        # Select rows in the random_tenser where mask is true. For those rows, copy the row from goal
        # and assign it to targetnode
        targetnode = torch.clone(samples)
        
        targetnode[mask] = goal[mask]
        

        # Now that we have the targetnode for each batch, we can move on to calculating what the next node is

        # we start by doing an 'unsqueeze' process on the targetnode. I don't really know how it works but this is
        # necessary for targetnode and tree_positions to be compatible for the subtraction
        targetnode_compat = targetnode.unsqueeze(1)

        # we then calculate the differenct between the tree_positions and targetnode_compat
        diff = targetnode_compat - tree_positions

        # now we have a difference tensor with the same size as tree_positions

        # next we square each element in the difference tensor
        squared_diff = diff**2

        # and then we take the square-root of the squared_diff tensor to get the distances
        distances = torch.sqrt(torch.sum(squared_diff, dim=2))

        # Create a mask based on nodecounts where any index in distances beyond nodecounts has its distance set to
        # zero

        # this defines the number of nodes as a column vector. Shape is (1, NMAX)
        node_indices = torch.arange(NMAX, device=device).unsqueeze(0)

        # we then define a mask to test whether node_indices its less that node_counts.unsqueeze(1). Shape is (B, NMAX)
        # node indices is a list from 0 to NMAX-1
        # node_counts defines how many nodes are currently in the tree
        # valid_mask ensures that empty unseen node spaces in node_indices (the buffer up to the max number of nodes) are not included
        # in the distance calculation. Instead the distances for those node indices are set to infinity
        valid_mask = node_indices < node_counts.unsqueeze(1)

        # apply the mask to distances to check indices
        distances[~valid_mask] = float('inf')

        # after calculating the distance we can then look for the index where the distance is the smallest
        # this is of size (batch_size, 1)
        nearest_indices = torch.argmin(distances, dim=1) # Shape is (batch_size,)
        batch_indices = torch.arange(batch_size, device=device) # Shape is (batch_size,)

        # next we find the node (in tree positions) that corresponds to that index
        nearnode = tree_positions[batch_indices, nearest_indices, :] # this should be of shape (batch_size, 1, 2 -->(x,y))

        # next we get the minimum distance
        min_dist = distances[batch_indices, nearest_indices]

        # next we calculate the new x and y coordinates of the next node
        nextnode = nearnode + (STEP_SIZE/min_dist).unsqueeze(1) * diff[batch_indices, nearest_indices,:]

        # Now that we have nextnode, we should then check the validity of next node by checking if it's in
        # free space and if it connects

        # first, check if the node is inFreespace()
        # we first define the node to be checked based on its x and y positions
        
        # right now, nextnode contains the coordinates for each batch
        # we can convert this tensor into a numpy array

        nextnode_cpu = nextnode.cpu().numpy()

        # freespace_mask_cpu is a numpy array
        freespace_mask_cpu = inFreespace(nextnode_cpu)

        # convert the numpy array to a tensor
        freespace_mask = torch.from_numpy(freespace_mask_cpu).to(device)

        # next, we have to check if the nearnode connects to the nextnode
        # we follow a similar procesure where we first convert the nearnode tensor into a numpy array
        nearnode_cpu = nearnode.cpu().numpy()

        # then we send the nearnode numpy array and the nextnode numpy array to the connectsTo function
        connects_mask_cpu = connectsTo(nearnode_cpu, nextnode_cpu)

        # convert the connects mask to a GPU tensor
        connects_mask = torch.from_numpy(connects_mask_cpu).to(device)

        # next we need to combine the masks to see, for each batch, whether the node found is valid
        next_valid_mask = freespace_mask & connects_mask

        # get the batches where the nextnode is valid
        valid_batches = torch.where(next_valid_mask)[0]

        # get the valid nextnodes
        valid_nextnodes = nextnode[valid_batches]

        # get only the valid nearest indices
        valid_nearest = nearest_indices[valid_batches]

        # call addtotree for only the valid batches and valid nodes
        addtotree(valid_batches, valid_nextnodes, valid_nearest)

        if valid_nextnodes.shape[0] == 0:
            continue
        
        # next I need to check whether the goal has been found

        # first make a mask to check whether the distance between the valid_nextnode and the goalnode
        # is within step_size

        # valid_nextnodes would be a subset of nextnodes which has shape valid_batches, (x,y)

        # Currently, the goalnode is of size (batch_size, (x,y)). Need to define valid_goal
        possible_goal = goal[valid_batches]

        # Find the Euclidean distance between the goal and each node in nextnode

        goal_diff = possible_goal - valid_nextnodes
        squared_goal_diff = goal_diff ** 2
        goal_distances = torch.sqrt(torch.sum(squared_goal_diff, dim=1))

        # Next we create a mesh to see whether the distance is within the step size
        print(f'Possible Goal: {possible_goal}')
        print(f'Valid Next Node: {valid_nextnodes}')
        print(f'Goal distances: {goal_distances}')
        dist_mask = goal_distances < STEP_SIZE

        # Next we must create another mesh that tests whether the valid nextnode connects to the 
        # goalnode

        # convert the tensor of valid nextnodes to numpy array
        valid_nearnode_cpu = valid_nextnodes.cpu().numpy()

        # conver the tensor of valid goals to a numpy array as well
        possible_goal_cpu = possible_goal.cpu().numpy()


        goal_connects_mask_cpu = connectsTo(valid_nearnode_cpu, possible_goal_cpu)

        # convert the goal connection mask to a tensor
        goal_connects_mask = torch.from_numpy(goal_connects_mask_cpu).to(device)

        # compare both masks
        print('Check mask types')
        print(dist_mask.dtype)
        print(f'dist_mask: {dist_mask}')
        print(goal_connects_mask.dtype)
        print(f'goal connects mask: {goal_connects_mask}')
        valid_goal_connects_mask = dist_mask & goal_connects_mask

        # get the batch indices where the goal can be added to the tree
        valid_goal_batches = torch.where(valid_goal_connects_mask)[0]

        valid_goal = possible_goal[valid_goal_connects_mask]
        
        # next we add the valid goal to the tree 
        # the parameters we need are the valid batch numbers, the valid goal nodes, and
        # the index of 'next node'

        goal_parents = node_counts[valid_goal_batches] - 1

        addtotree(valid_goal_batches, valid_goal, goal_parents)

        # Set the batch to inactive
    
        # the last thing is to check the following criterion for whether to stop a batch
        # 1. The goal has been found
        active_batches[valid_goal_batches] = False
        
        # Increment the number of steps. Recall the following step tensor definition
        # step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        step_counts += 1

        # Test to see if the step counts for any batch has exceeded the maximum steps
        step_mask = step_counts >= SMAX

        # Test to see if the node counts for any natch has exceeded the maximum nodes
        node_mask = node_counts >= NMAX

        # Create the overall expired mask
        expired_mask = step_mask | node_mask

        # Apply the expired mask to active batches
        active_batches[expired_mask] = False

        # And then the last thing is that I should break/end the loop if all the batches
        # have been stopped

        if not active_batches.any():
            if expired_mask == True:
                print(f'Process Aborted at Node Count = {node_counts} and \nStep Count = {step_counts}. No path found')
                return None
            break

    # We're finally out of the loop!!!

    # Now we need to create the path
    
    # This path re-contruction is done not in GPU because it can't be done simultaneously
    # As a result, this means we need to convert nodecounts, tree_parents, and tree_positions
    # to numpy arrays

    node_counts_cpu = node_counts.cpu().numpy()
    tree_parents_cpu = tree_parents.cpu().numpy()
    tree_positions_cpu = tree_positions.cpu().numpy()
    goal_indices = node_counts_cpu - 1

    # Let's send all this stuff to a function outside of the rrt called buildPath
    all_paths = buildPath(goal_indices, tree_parents_cpu, tree_positions_cpu, batch_size)
    return all_paths


def buildPath(goal_idx, node_parents, node_positions, batch_size):

    # goal_idx = np_goal_idx.tolist()
    # node_parents = np_node_parents.tolist()
    # node_positions = np_node_positions.tolist()

    
    print('Now building the path')
    path = []
    # loop through each batch
    for batch_num in range(batch_size):
        current_path = []
        idx = goal_idx[batch_num]

        # Skip if no path was found:
        if idx < 0 or node_parents[batch_num, idx] == -1 and idx != 0:
            path.append(None)
            continue

        while idx != -1:
            # the parent node hasn't been encountered
            # add the node to the path
            current_path.append(node_positions[batch_num, idx])
            idx = node_parents[batch_num, idx]

        current_path.reverse()
        tup_list = [tuple(float(x) for x in point) for point in current_path]
        node_list = [Node(x,y) for (x,y) in tup_list]
        path.append(node_list)
        print(node_list)
    return path
            
 
# Post process the path.
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1

# MAIN
def main():
    print('Running with step size ', STEP_SIZE, ' and up to ', NMAX, ' nodes.')

    # Create the figure
    visual = Visualization()

    # Create the start and goal nodes
    startnode = Node(xstart, ystart)
    goalnode = Node(xgoal, ygoal)

    # Visualize the start and goal nodes
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode, color='purple', marker='o')
    visual.show('Showing basic world') 

    # Call the RRT function
    print('Running RRT')
    paths = rrt(startnode, goalnode, visual)

    # If unable to connect path, note this
    if not paths:
        visual.show('NO PATHS FOUND')
        return
    
    # Otherwise, show the path created
    colors = ['r', 'b', 'g', 'y']
    index = 0
    for path in paths:
        visual.drawPath(path, color=colors[index], linewidth=1)
        visual.show(f'Showing the raw path for batch {index + 1}')
        index += 1


    # Post-process the path
    # PostProcess(path)
        
    # Show the post-processed path
    # visual.drawPath(paths, color='b', linewidth=2)
    # visual.show('Showing the post-processed path')

if __name__ == "__main__":
    main()