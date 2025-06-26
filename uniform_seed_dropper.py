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

STEP_SIZE = 0.25
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
wall1   = LineString([[xmin, yB], [xC, yB]])
wall2   = LineString([[xD, yB], [xmax, yB]])
wall3   = LineString([[xB, yC], [xC, yC]])
wall4   = LineString([[xC, yC], [xC, ymax]])
wall5   = LineString([[xC, yB],[xC, yA]])
wall6   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])

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

batch_size = 4
num_seeds = 3

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
    
def place_nodes():

    # here we uniformly sample the grid

    # probably update the samples tensor to be of shape (batch_size, num_seeds_2)

    samples = torch.empty((batch_size, num_seeds, 2), device=device)

    samples[:,:,0] = torch.rand()


    samples = torch.empty((batch_size, 2), device=device)
    samples[:, 0] = torch.rand(batch_size, device=device) * (xmax - xmin) + xmin # at most xmax at least xmin
    samples[:, 1] = torch.rand(batch_size, device=device) * (ymax - ymin) + ymin # at most ymax at least ymin




    print(samples[:,0])
    print(samples[:,1])

