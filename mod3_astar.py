# INCREASING THE WIDTH OF THE ROBOT TO TWO SQUARES.
# RE-DEFINE COORDINATES, NEIGHBORS, HEURISTIC
# NO ORIENTATION TRACKING/COSTS APPLIED

# IMPORT MODULES
import bisect # for sorting nodes on Deck
from math import inf # idk what this is for yet but it was included
from visualgrid import VisualGrid # helper module for visualizing 2D grid
import math

# DEFINE THE GRID
grid = ['##############',
        '#            #', 
        '#     ##     #',
        '# LR    ##lr #',
        '#            #',
        '##############']


# THIS GRID WORKS
# grid = ['##########',
#         '#        #',
#         '# LR  #lr#',
#         '##########']

# DEFINE COLORS :D
# using pastel color scheme because why not?
WHITE = [0.980, 0.973, 0.965]
GREEN = [0.816, 0.914, 0.753]
BLUE = [0.745, 0.867, 0.945]
PINK = [0.965, 0.757, 0.698]
TAN = [0.914, 0.788, 0.667]
ROSE = [0.965, 0.722, 0.816]
GOLD = [0.973, 0.773, 0.486]
RED = [0.929, 0.525, 0.596]
BLACK = [0.000, 0.000, 0.000]

# DEFINE THE NODE CLASS
# THIS CLASS STORES location, neighbors, cost, parents, status
class Node:
    def __init__(self, row_left, col_left, row_right, col_right):
        self.row_left = row_left
        self.row_right = row_right
        self.col_left = col_left
        self.col_right = col_right

        self.neighbors = []

        self.parent = None
        self.creach = inf
        self.cost = inf

        self.seen = False
        self.done = False

    def distance(self, other):
        # takes the midpoint of self and midpoint of other and then takes the
        # difference to get the distance

        self_midrow = 0.5*(self.row_left + self.row_right)
        self_midcol = 0.5*(self.col_left + self.col_right)

        other_midrow = 0.5*(other.row_left + other.row_right)
        other_midcol = 0.5*(other.col_left + other.col_right)

        return abs(self_midrow - other_midrow) + abs(self_midcol - other_midcol)
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __str__(self):
        return ("(%2d,%2d,%2d,%2d)" % (self.row_left, self.col_left, self.row_right, self.col_right))
    
    def __repr__(self):
        return("<Node %s, %7s, cost %f>" %
               (str(self),
                "done" if self.done else "seen" if self.seen else "unknown",
                self.cost))

def costtogoest(node, goal):
    # Let's use the Chebyshev distance metric
    node_midrow = 0.5*(node.row_left + node.row_right)
    node_midcol = 0.5*(node.col_left + node.col_right)

    goal_midrow = 0.5*(goal.row_left + goal.row_right)
    goal_midcol = 0.5*(goal.col_left + goal.col_right)

    return max(abs(node_midrow - goal_midrow), abs(node_midcol - goal_midcol))

def planner(start, goal, show = None):
    start.seen = True
    start.creach = 0
    start.cost = 0 + costtogoest(start, goal)
    start.parent = None
    onDeck = [start]

    print("Starting the processing...")
    while True:
        if show:
            show()

        if not (len(onDeck) > 0):
            return None
        
        node = onDeck.pop(0)

        node.done = True

        if goal.done:
            break

        for neighbor in node.neighbors:
            if neighbor.done:
                continue

            creach = node.creach + 1

            if neighbor.seen:
                if neighbor.creach <= creach:
                    continue
                else:
                    onDeck.remove(neighbor)

            neighbor.seen = True
            neighbor.creach = creach
            neighbor.cost = creach + costtogoest(neighbor, goal)
            neighbor.parent = node
            bisect.insort(onDeck, neighbor)

    path = [goal]
    while path[0].parent:
        path.insert(0, path[0].parent)
    return path

if __name__== "__main__":

    # Crate the back-end of the grid
    rows = len(grid)
    cols = max([len(line) for line in grid])

    # Set up the visual grid
    visual = VisualGrid(rows, cols)


    # create nodes based on free grid spaces
    nodes = []

    second_node = [(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(-1,1),(1,-1),(1,1)]
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == '#':
                visual.color(row, col, BLACK)
            else:
                for (dr,dc) in second_node:
                    if grid[row+dr][col+dc] != '#':
                        nodes.append(Node(row,col,row+dr,col+dc))
    
    for node in nodes:
        for node in nodes:
            # check for node type

            if math.sqrt((node.row_left - node.row_right)**2 + (node.col_left - node.col_right)**2) == 1:
                # the node has squares that are adjacent
                for (Ldr, Ldc, Rdr, Rdc) in [(0,-1,0,-1), (0,1,0,1), (-1,0,-1,0), (1,0,1,0),
                         (0,0,-1,0), (0,0,1,0), (-1,0,0,0), (1,0,0,0)]:
                    others = [n for n in nodes
                              if (n.row_left, n.col_left, n.row_right, n.col_right) == (node.row_left + Ldr, node.col_left + Ldc, node.row_right + Rdr, node.col_right + Rdc)]   
                    if len(others) > 0:
                        node.neighbors.append(others[0]) 
            
            elif node.row_left < node.row_right:
                # the node has squares that are left-forward diagonal
                for (Ldr, Ldc, Rdr, Rdc) in [(-1,-1,-1,-1), (1,1,1,1), (-1,1,-1,1), (1,-1,1,-1),
                         (0,0,-1,0), (0,0,0,-1), (0,1,0,0), (1,0,0,0)]:
                    others = [n for n in nodes
                              if (n.row_left, n.col_left, n.row_right, n.col_right) == (node.row_left + Ldr, node.col_left + Ldc, node.row_right + Rdr, node.col_right + Rdc)]   
                    if len(others) > 0:
                        node.neighbors.append(others[0]) 

            elif node.row_left > node.row_right:
                # the node has squares that are right-forward diagonal
                for (Ldr, Ldc, Rdr, Rdc) in [(-1,-1,-1,-1), (1,1,1,1), (-1,1,-1,1), (1,-1,1,-1),
                         (0,0,1,0), (0,0,0,-1), (0,1,0,0), (-1,0,0,0)]:
                    others = [n for n in nodes
                              if (n.row_left, n.col_left, n.row_right, n.col_right) == (node.row_left + Ldr, node.col_left + Ldc, node.row_right + Rdr, node.col_right + Rdc)]   
                    if len(others) > 0:
                        node.neighbors.append(others[0])

    #         print(f'Number of neighbours = {len(node.neighbors)}')
    # print(f'Number of nodes = {len(nodes)}')
    start = [n for n in nodes if (grid[n.row_left][n.col_left] in 'L') and (grid[n.row_right][n.col_right] in 'R')][0]
    goal = [n for n in nodes if (grid[n.row_left][n.col_left] in 'l') and (grid[n.row_right][n.col_right] in 'r')][0]
    visual.write(start.row_left, start.col_left, 'L')
    visual.write(start.row_right, start.col_right, 'R')
    visual.write(goal.row_left, goal.col_left, 'l')
    visual.write(goal.row_right, goal.col_right, 'r')
    visual.show(wait="Hit return to start")

    def show(wait=0.005):
        for node in nodes:
            if node.done:   visual.color(node.row_left, node.col_left, TAN)
            elif node.seen: visual.color(node.row_left, node.col_left, GREEN)
            else:           visual.color(node.row_left, node.col_left, BLUE)
        visual.show(wait)

    path = planner(start, goal, show)

    unknown = len([n for n in nodes if not n.seen])
    processed = len([n for n in nodes if n.done])
    ondeck = len(nodes) - unknown - processed
    print("Solution cost %f" % goal.cost)
    print("%3d states fully processed" % processed)
    print("%3d states still pending" % ondeck)
    print("%3d states never reached" % unknown)

    if not path:
        print("UNABLE TO FIND A PATH")
    else:
        print("Marking the path")
        for node in path:
            visual.color(node.row_left, node.col_left, RED)
            print(node.row_left, node.col_left, node.row_right, node.col_right)
        visual.show()

    input("Hit return to end")