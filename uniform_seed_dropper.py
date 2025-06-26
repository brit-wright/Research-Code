import torch

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
import cProfile

STEP_SIZE = 0.5
NMAX = 50
SMAX = 50

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
# wall1   = LineString([[xmin, yB], [xC, yB]])
# wall2   = LineString([[xD, yB], [xmax, yB]])
# wall3   = LineString([[xB, yC], [xC, yC]])
# wall4   = LineString([[xC, yC], [xC, ymax]])
# wall5   = LineString([[xC, yB],[xC, yA]])
# wall6   = LineString([[xC, ymin], [xB, yA]])
# bonus   = LineString([[xD, yC], [xE, yC]])

device='cuda'
# twall1   = [[xmin, yB], [xC, yB]]
# twall2   = [[xD, yB], [xmax, yB]]
# twall3   = [[xB, yC], [xC, yC]]
# twall4   = [[xC, yC], [xC, ymax]]
# twall5   = [[xC, yB],[xC, yA]]
# twall6   = [[xC, ymin], [xB, yA]]
# tbonus   = [[xD, yC], [xE, yC]]
# twalls = [twall1, twall2, twall3, twall4, twall5, twall6, tbonus]
# wall_coords = torch.tensor([[[twall[0][0],twall[0][1]], [twall[1][0],twall[1][1]]] for twall in twalls], dtype=torch.float, device=device)
# W = wall_coords.shape[0]

# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
# walls = prep(MultiLineString([outside, wall1, wall2, wall3, wall4, wall5, wall6, bonus]))
# walls = prep(MultiLineString([outside, wall1, wall2, wall3]))

# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
walls = prep(MultiLineString([outside]))

# Define the start/goal states (x, y, theta) of the mattress
(xstart, ystart) = (xA, yD)
(xgoal, ygoal) = (5, 5)

batch_size = 4
num_seeds = 10

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

def place_nodes():

    # here we uniformly sample the grid
    # need to introduce a mask to make sure seeds are not places where 
    # obstacles are

    # it's just the same infreespace mask lol

    samples = torch.empty((batch_size, num_seeds, 2), device=device)
    samples[:,:,0] = torch.rand(batch_size, num_seeds, device=device) * (xmax - xmin) + xmin
    samples[:,:,1] = torch.rand(batch_size, num_seeds, device=device) * (ymax - ymin) + ymin
    return samples

def rrt(start, goalnode, visuals):
    
    # Leaving many more comments because I'm genuinely confused lol

    ################### LET'S START BY DEFINING THE IMPORTANT VARIABLES FOR THIS PROBLEM ####################

    # first we need to indicate that we're changing the processing/device from CPU to GPU (cuda)
    device = torch.device('cuda')   

    # Batch Size is basically the number of RRTs I want to have running in parallel
    batch_size = 4

    # Now I'm defining node counts which basically tells us, how many nodes are in each tree
    # where it's one tree per batch
    node_counts = torch.ones(batch_size, num_seeds, dtype=torch.long, device=device)

    # Defining the start and the goal node as tensors containing the x and y positions of the
    # nodes
    
    goal = torch.tensor([[goalnode.x, goalnode.y]], dtype=torch.float, device=device).repeat(batch_size, num_seeds, 1)

    # Next, I'm creating the tree_parents tensor of size batch_size, NMAX
    # How it works is that for each nth node (which is defined based on the location in NMAX)
    # It stores the index of that node's parent. Currently initialized to -1 meaning that 
    # None is assigned to parents.
    tree_parents = torch.full((batch_size, num_seeds, NMAX), -1, dtype=torch.long, device=device)

    # Next we define the tree. For each batch, the tree will store the (x,y) positions of the 
    # nth node that has been processed. The nth node is defined based on the location of the 
    # node in NMAX

    # Tree positions is also initialized for the zeroth node as the startnode. This is equivalent
    # to saying tree = [startnode]

    tree_positions = torch.zeros((batch_size, num_seeds, NMAX, 2), device=device) # has shape (batch_size, num_seeds, NMAX, 2)
    # tree_positions[:, :, 0, 0] = startnode.x
    # tree_positions[:, :, 0, 1] = startnode.y
    tree_positions[:, :, 0, :] = start

    iter = 0
    ########### NOW LET'S GET TO WRITING THE addtotree FUNCTION ##################
    def addtotree(valid_nearnodes, valid_batches, valid_seeds, valid_nextnodes, nearest_indices, visuals):
        # start by assigning the parent of the new_node to be the nearnode
        # print(f'Shape of tree_parents: {tree_parents.shape}')
        # print(f'Shape of valid_nearnodes: {valid_nearnodes.shape}')
        # print(f'Shape of valid_batches: {valid_batches.shape}')
        # print(f'Shape of valid_seeds: {valid_seeds.shape}')
        # print(f'Shape of valid_nextnodes: {valid_nextnodes.shape}')
        # print(f'Shape of nearest_indices: {nearest_indices.shape}')
        # print(f'Shape of node_counts[valid_batches]: {node_counts[valid_batches].shape}')
        # print(f'Shape of node_counts: {node_counts.shape}')
        tree_parents[valid_batches, valid_seeds, node_counts] = nearest_indices

        # add the new node to the tree
        tree_positions[valid_batches, valid_seeds, node_counts, :] = valid_nextnodes

        # increment the node cout
        next_node_index = node_counts[valid_batches, valid_seeds]
        node_counts[valid_batches, valid_seeds] += 1

        # print(valid_batches.shape[0])
        # print(valid_seeds.shape[1])

        for b in range(valid_batches.shape[0]):
            for s in range(valid_seeds.shape[1]):
                batch = valid_batches[b][0].item()
                seed = valid_seeds[0][s]

                old_x, old_y = valid_nearnodes[batch][seed]
                new_x, new_y = valid_nextnodes[batch][seed]

                oldnode = Node(old_x.item(), old_y.item())
                newnode = Node(new_x.item(), new_y.item())

                # Draw on the batch-specific visualizer
                visuals[batch].drawEdge(oldnode, newnode, color='g', linewidth=1)
                visuals[batch].show()


    ########### OKAY LET'S GO INTO THE ACTUAL LOOP LOGIC #####################
    # Also define the number of steps that have been made, since we have a constraint on the number
    # steps that are supposed to be done for each tree
    step_counts = torch.zeros(batch_size, num_seeds, dtype=torch.long, device=device)
    
    
    # ALSO DEFINE THE PROBABILITY
    p = 0.3

    # DEFINE THE FLAGS/STOPPERS
    active_batches = torch.ones(batch_size, num_seeds, dtype=torch.bool, device=device)
    ######### ENTER THE WHILE LOOP LOL #######################
    while True:
        iter += 1    

        # Define the samples tensor which does uniform stampling
        targetnode = torch.empty((batch_size, num_seeds, 2), device=device) # has shape (batch_size, num_seeds, 2)
        targetnode[:, :, 0] = torch.rand(batch_size, num_seeds, device=device) * (xmax - xmin) + xmin
        targetnode[:, :, 1] = torch.rand(batch_size, num_seeds, device=device) * (ymax - ymin) + ymin

        # originally the shape of targetnode is (batch_size, num_seeds)

        # Now that we have the targetnode for each batch, we can move on to calculating what the next node is

        # we start by doing an 'unsqueeze' process on the targetnode. I don't really know how it works but this is
        # necessary for targetnode and tree_positions to be compatible for the subtraction
        targetnode_compat = targetnode.unsqueeze(2) # has shape (batch_size, num_seeds, 1, 2)

        # we then calculate the differenct between the tree_positions and targetnode_compat
        diff = targetnode_compat - tree_positions
        # print(f'Shape of diff is: {diff.shape}')
        # shape is (4, 3, 50, 2)

        # now we have a difference tensor with the same size as tree_positions

        # next we square each element in the difference tensor
        squared_diff = diff**2

        # and then we take the square-root of the squared_diff tensor to get the distances
        # for inactive batches, the distance is infinity
        distances = torch.sqrt(torch.sum(squared_diff, dim=-1)) # shape is (batch_size, num_seeds, NMAX)
        # print(f'Shape of distances: {distances.shape}')

        # distances has shape [4, 3, 2] --> 4 for the number of batches, 3 for the number of seeds, and 2 for...

        # Create a mask based on nodecounts where any index in distances beyond nodecounts has its distance set to
        # zero

        # this defines the number of nodes as a column vector. Shape is (NMAX, 1)
        node_indices = torch.arange(NMAX, device=device).view(1, 1, -1) # Shape is (1, 1, NMAX)

        # we then define a mask to test whether node_indices is less that node_counts.unsqueeze(1). Shape is (B, NMAX)


        # node_counts has shape (batch_size, num_seeds)
        # node_indices has shape (1, NMAX)

        # node indices is a list from 0 to NMAX-1
        # node_counts defines how many nodes are currently in the tree
        # valid_mask ensures that empty unseen node spaces in node_indices (the buffer up to the max number of nodes) are not included
        # in the distance calculation. Instead the distances for those node indices are set to infinity
        
        masked_node_counts = node_counts.clone() # has shape of node_counts: batch_size, num_seeds
        
        # valid mask is used to say which values should be considered finding the smallest distance by taking 
        # into account whether the node number has been seen (we don't calculate distance for nodes that don't
        # yet exist). Masked node counts has 0 where the batch is inactive an integer where the batch is active

        # So for an inactive node, node_indices < masked_node_counts will always be False for inactive batches 
        # and so the distance will be set to infinity for all nodes in that batch

        # It also retains its original function where in active batches, any node count past the actual number of 
        # nodes in the tree is set to False

        # resulting shape of valid_mask is (batch_size, num_seeds, NMAX)
        valid_mask = node_indices < node_counts.unsqueeze(2) # comparing the shape (1, 1, NMAX) to (batch_size, num_seeds, 1) --> checks for each batch_size whether the node should be processed

        # apply the mask to distances to check indices
        distances[~valid_mask] = float('inf')

        # after calculating the distance we can then look for the index where the distance is the smallest
        # this is of size (batch_size, 1)
        nearest_indices = torch.argmin(distances, dim=2) # Shape is (batch_size,)
        # print(f'Shape of nearest_indices: {nearest_indices.shape}')


        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(batch_size, num_seeds) # Shape is (batch_size, seed_num)
        # print(f'batch indices: {batch_indices.shape}')
        
        seed_indices = torch.arange(num_seeds, device=device).view(1,-1).expand(batch_size, num_seeds) # Shape is (batch_size, seed_num)
        # print(f'seed indices: {seed_indices.shape}')

        # next we find the node (in tree positions) that corresponds to that index
        # print(f'shape of tree positions: {tree_positions.shape}')
        nearnode = tree_positions[batch_indices, seed_indices, nearest_indices] # this should be of shape (batch_size, num_seeds, 1, 2 -->(x,y))

        # next we get the minimum distance
        min_dist = distances[batch_indices, seed_indices, nearest_indices]

        # next we calculate the new x and y coordinates of the next node
        # print(f'Shape of nearnode: {nearnode.shape}')
        # print(f'Shape of step_size/min_dist: {(STEP_SIZE/min_dist).shape}')
        # print(f'Shape of diff thing: {(diff[batch_indices[:,None], seed_indices[None,:], nearest_indices]).shape}')
        # print(f'Shape of diff: {diff.shape}')
        # print((batch_indices[:,None]).shape)
        # print((seed_indices[None,:]).shape)
        nextnode = nearnode + (STEP_SIZE/min_dist).unsqueeze(2) * diff[batch_indices, seed_indices, nearest_indices]

        addtotree(nearnode, batch_indices, seed_indices, nextnode, nearest_indices, visuals)

        # Test to see if the step counts for any batch has exceeded the maximum steps
        step_mask = step_counts >= SMAX

        # Test to see if the node counts for any batch has exceeded the maximum nodes
        node_mask = node_counts >= NMAX

        # Create the overall expired mask
        expired_mask = step_mask | node_mask

        # Apply the expired mask to active batches
        active_batches[expired_mask] = False

        # And then the last thing is that I should break/end the loop if all the batches
        # have been stopped
        if False in active_batches:
            # print(active_batches)
            pass
        if not active_batches.any():
            if (step_counts >= SMAX).all() | (node_counts >= NMAX).all():
                # print(f'Process Aborted at Node Count = {node_counts} and \nStep Count = {step_counts}. No path found')
                return None
            break

    # We're finally out of the loop!!!

    # Now we need to create the path
    
    # This path re-contruction is done not in GPU because it can't be done simultaneously
    # As a result, this means we need to convert nodecounts, tree_parents, and tree_positions
    # to numpy arrays
    tf = time.time()
    print(node_counts, step_counts)
    # print(f'Stuck counter: {stuck_counter}')
    node_counts_cpu = node_counts.cpu().numpy()
    tree_parents_cpu = tree_parents.cpu().numpy()
    tree_positions_cpu = tree_positions.cpu().numpy()
    goal_indices = node_counts_cpu - 1
    # all_goal_batches_cpu = all_goal_batches.cpu().numpy()
    # Let's send all this stuff to a function outside of the rrt called buildPath
    # all_paths = buildPath(goal_indices, tree_parents_cpu, tree_positions_cpu, batch_size, all_goal_batches_cpu)

# Write out the main function here

# MAIN
def main():
    print('Running with step size ', STEP_SIZE, ' and up to ', NMAX, ' nodes.')

    # Create the figure
    visuals = [Visualization(b) for b in range(batch_size)]

    
    start = place_nodes() # returns one startnode per batch per seed

    for b in range(batch_size):
        for s in range(num_seeds):
            x, y = start[b, s]
            visuals[b].drawNode(Node(x.item(), y.item()), color='orange', marker='o')
        visuals[b].show('Showing basic world') 

    # Create the start and goal nodes
    startnode = Node(xstart, ystart)
    goalnode = Node(xgoal, ygoal)

    # Visualize the start and goal nodes
    # for visual in visuals:
    #     visual.drawNode(startnode, color='orange', marker='o')
    #     visual.drawNode(goalnode, color='purple', marker='o')
    #     visual.show('Showing basic world') 


    # Call the RRT function
    print('Running RRT')
    rrt(start, goalnode, visuals)

    ans = int(input('Enter 0 to end.'))
    
if __name__ == "__main__":
    main()
    # cProfile.run('main()')