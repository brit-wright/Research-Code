# THIS IS THE BASIC A* PLANNING CODE THAT DOESN'T CONSIDER
# ROBOT ORIENTATION.

# IMPORT MODULES
import bisect # for sorting nodes on Deck
from math import inf # idk what this is for yet but it was included
from visualgrid import VisualGrid # helper module for visualizing 2D grid


# DEFINE THE GRID
# grid = ['####################',
#         '#                  #',
#         '#                  #', 
#         '#   ########       #',
#         '#          ##      #',
#         '#   S       ##G    #',
#         '#            ####  #',
#         '#                  #',
#         '#                  #',
#         '####################']

grid = ['####################',
        '#                  #',
        '#                  #', 
        '#   ######  #      #',
        '#  ##  #   #  #    #',
        '#   S # # # # G    #',
        '#      #   # ####  #',
        '####################']

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
    def __init__(self, row, col):
        self.row = row
        self.col = col

        self.neighbors = []

        self.parent = None
        self.creach = inf
        self.cost = inf

        self.seen = False
        self.done = False

    def distance(self, other):
        return abs(self.row - other.row) + abs(self.col - other.col)
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __str__(self):
        return ("(%2d,%2d)" % (self.row, self.col))
    
    def __repr__(self):
        return("<Node %s, %7s, cost %f>" %
               (str(self),
                "done" if self.done else "seen" if self.seen else "unknown",
                self.cost))

def costtogoest(node, goal):
    return max(abs(node.row - goal.row), abs(node.col - goal.col))

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

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == '#':
                visual.color(row, col, BLACK)
            else:
                nodes.append(Node(row, col))

    for node in nodes:
        for (dr, dc) in [(-1,0), (-1,1), (0,1), (1,1),
                         (1,0), (1,-1), (0,-1), (-1,-1)]:
            others = [n for n in nodes
                      if (n.row, n.col) == (node.row+dr, node.col+dc)]
            if len(others) > 0:
                node.neighbors.append(others[0])

    start = [n for n in nodes if grid[n.row][n.col] in 'Ss'][0]
    goal = [n for n in nodes if grid[n.row][n.col] in 'Gg'][0]
    visual.write(start.row, start.col, 'S')
    visual.write(goal.row, goal.col, 'G')
    visual.show(wait="Hit return to start")

    def show(wait=0.005):
        for node in nodes:
            if node.done:   visual.color(node.row, node.col, TAN)
            elif node.seen: visual.color(node.row, node.col, GREEN)
            else:           visual.color(node.row, node.col, BLUE)
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
            visual.color(node.row, node.col, RED)
        visual.show()

    input("Hit return to end")