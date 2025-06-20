import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
from vandercorput       import vandercorput

# PARAMETERS
STEP_SIZE = 0.25 # chose a static step size
SMAX = 50000 # maximum number of step attempts
NMAX = 20000 # maximum number of nodes

# Define the list of obstacles/objects as well as star/goal

# Define the vertices of the shapes that will be drawn to define the objects/obstacles
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
bonus   = LineString([[xD, yC], [xE, yC]])

# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
# walls = prep(MultiLineString([outside, wall1, wall2, wall3, bonus]))
walls = prep(MultiLineString([outside, wall1, wall2, wall3]))

# Define the start/goal states (x, y, theta) of the mattress
(xstart, ystart, tstart) = (xA, yD, pi/2)
(xgoal, ygoal, tgoal) = (xA, yA, 0)

# Define the mattress dimensions
L = 7
W = 1

# Utilities: Angle Wrapping and Visualization

# Angle Wrapping Utility
def wrap(angle, fullrange):
    return angle - fullrange * round(angle/fullrange)

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
        plt.plot(*node.mattress.exterior.xy, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot([head.x, tail.x], [head.y, tail.y], *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            n = ceil(path[i].distance(path[i+1]) / STEP_SIZE)
            for j in range(n):
                node = path[i].intermediate(path[i+1], j/n)
                self.drawNode(node, *args, **kwargs)
        self.drawNode(path[-1], *args, **kwargs)

# NODE DEFINITION

class Node:
    def __init__(self, x, y, theta):
        # Define a parent 
        self.parent = None

        # Precompute sin/cos functions
        c = cos(theta)
        s = sin(theta)

        # Define/remember the state/coordinates (x,y,theta) of the node
        self.x = x
        self.y = y
        self.s = L/2 * (s*c)
        self.c = L/2 * (c*c - s*s)/2
        self.t = theta

        # Pre-compute the mattress box (for collision detection and drawing)
        self.mattress = Polygon([[x + L/2*c + W/2*s, y + L/2*s - W/2*c],
                                 [x + L/2*c - W/2*s, y + L/2*s + W/2*c],
                                 [x - L/2*c - W/2*s, y - L/2*s + W/2*c],
                                 [x - L/2*c + W/2*s, y - L/2*s - W/2*c],
                                 [x + L/2*c + W/2*s, y + L/2*s - W/2*c]])
        
    # Node Utilities
    # To print the node
    def __repr__(self):
        return("<Node %5.2f,%5.2f @ %5.2f>" % (self.x, self.y, self.t))
    
    # Compute/create an intermediate node for checking the local planner
    def intermediate(self, other, alpha):
        return Node(self.x + alpha *     (other.x - self.x),
                self.y + alpha *     (other.y - self.y),
                self.t + alpha * wrap(other.t - self.t, pi))
    
    # Return a tuple of coordinates to compute Euclidean distances
    def coordinates(self):
        return(self.x, self.y, self.s, self.c)
    
    # Compute the relative distance Euclidean distance to another node
    def distance(self, other):
        return dist(self.coordinates(), other.coordinates())
    
    # Collision functions:
    # Check whether in free space
    def inFreespace(self):
        return walls.disjoint(self.mattress)

    # Check the local planner - whether this node connects to another node

    def connectsTo(self, other):
        for delta in vandercorput(STEP_SIZE / self.distance(other)):
            if not self.intermediate(other, delta).inFreespace():
                return False
        return True

# RRT Functions

def rrt(startnode, goalnode, visual):
    startnode.parent = None
    tree = [startnode]

    # Add a new node to the tree
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)

        # Visualize the new node
        visual.drawEdge(oldnode, newnode, color = 'g', linewidth = 1)
        visual.show()

    steps = 0
    while True:
        # Select a random node using a uniform distribution over space
        targetnode = Node(random.uniform(xmin + W/2, xmax - W/2), random.uniform(ymin + L/2, ymax - L/2), random.uniform(-pi/2, pi/2))

        # Determine the distances from each node in the RRT tree to the targetnode
        distances = np.array([node.distance(targetnode) for node in tree])
        index   = np.argmin(distances) # returns the index of the node (tree position) that is closest to the targetnode
        nearnode = tree[index] # finds the closest node in the tree to the targetnode
        d = distances[index] # the distance between the nearnode and the target node

        # Get the nextnode (a node that is between the nearnode and the targetnode)
        x_next = nearnode.x + STEP_SIZE/d * (targetnode.x - nearnode.x)
        y_next = nearnode.y + STEP_SIZE/d * (targetnode.y - nearnode.y)

        # How to get the angle (theta_next)?
        
        # For now, just implement by taking the average between the nearnode and the targetnode
        theta_next = (nearnode.t + targetnode.t)/2
        
        nextnode = Node(x_next, y_next, theta_next)

        # Check that nextnode is valid
        if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
            addtotree(nearnode, nextnode)

            # Check for connection to goal if nextnode is within a stepsize
            if (nextnode.distance(goalnode) <= STEP_SIZE) and (nextnode.connectsTo(goalnode)):
                addtotree(nextnode, goalnode)
                break

        # Check whether step/node criterion met
        steps += 1

        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            return None
        
    # Create the path
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
          (steps, len(tree)))
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
    startnode = Node(xstart, ystart, tstart)
    goalnode = Node(xgoal, ygoal, tgoal)

    # Visualize the start and goal nodes
    visual.drawNode(startnode, color='orange', linewidth=3)
    visual.drawNode(goalnode, color='purple', linewidth=3)
    visual.show('Showing basic world') 

    # Call the RRT function
    print('Running RRT')
    path = rrt(startnode, goalnode, visual)

    # If unable to connect path, note this
    if not path:
        visual.show('UNABLE TO FIND A PATH')
        return
    
    # Otherwise, show the path created
    visual.drawPath(path, color='r', linewidth=1)
    visual.show('Showing the raw path')


    # Post-process the path
    PostProcess(path)
        
    # Show the post-processed path
    visual.drawPath(path, color='b', linewidth=2)
    visual.show('Showing the post-processed path')

if __name__ == "__main__":
    main()