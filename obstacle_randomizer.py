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

while ((abs(x1 - x2) <= 15 or abs(y1 - y2) <= 15) or (abs(x1 - x2) > 40 or abs(y1 - y2) > 40)):
    x2 = int(random.random() * 100)
    y2 = int(random.random() * 100)

print(x1, x2, y1, y2)
print(abs(x1 - x2))
print(abs(y1 - y2))


xmin = min(x1, x2)
xmax = max(x1, x2)
ymin = min(y1, y2)
ymax = max(y1, y2)


(xA, xB, xC, xD, xE) = ( 5, 12, 15, 18, 21)
(yA, yB, yC, yD)     = ( 5, 10, 12, 15)

xlabels = (xmin, xA, xB, xC, xD, xE, xmax)
ylabels = (ymin, yA, yB, yC, yD,     ymax)

tl = [xmin, ymax]
tr = [xmax, ymax]
bl = [xmin, ymin]
br = [xmax, ymin]

print(tl, tr, bl, br)

outside = LineString([bl, br, tr, tl, bl])
walls = prep(MultiLineString([outside]))

########################## INNER OBSTACLES #########################






















# Visualize it!

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

    batch_size = 4
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