# THIS VERSION OF THE A-STAR CODE SOLVES WITH COORDINATE OF X, Y, AND THETA, AND DOES SO USING AN 
# AUTO-GENERATED, USER-ALTERED GRID

# Creating a random grid generator based on user information.
# Will check how well A*star runs with it

# THIS IS A MODIFIED VERSION OF THE BASIC A* CODE THAT WILL CONSIDER THE
# ROBOT'S ORIENTATION

# IMPORT MODULES
import bisect # for sorting nodes on Deck
from math import inf # idk what this is for yet but it was included
from visualgrid import VisualGrid # helper module for visualizing 2D grid
import math
import random

# DEFINE COLORS :D
# using pastel color scheme because why not?
WHITE = [1.000, 1.000, 1.000]
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
        self.cturn = inf
        self.cost = inf
        self.angle = 0

        self.seen = False
        self.done = False

        self.orient = None #other postions are 'n', 's', 'e', 'w'

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

def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = dot/(mag1 * mag2)
    angle = math.acos(max(-1, min(1, cos_theta)))
    return math.floor(math.degrees(angle))

def costtoturn(node, neighbor):
    current = node.orient
    row_diff = neighbor.row - node.row
    col_diff = neighbor.col - node.col
    new_direction = (row_diff, col_diff)

    angle = angle_between(current, new_direction)
    if angle == 0:
        return 0, angle
    elif angle == 180:
        return 2, angle
    elif angle == 90:
        return 1, angle
    elif angle == 45:
        return 0.5, angle
    elif angle == 135:
        return 1.5, angle
    else:
        return 1, angle # fallback or general small turn cost
    
def add_obstacles():
    additions = []
    add = 1
    while add == 1:
        row = int(input('Enter the row you want to add an obstacle to '))
        col = int(input('Enter the column you want to add an obstacle to '))
        additions.append((row, col))
        add = int(input('Enter 1 to continue adding obstacles. Otherwise, enter 0. '))
    return additions

def remove_obstacles():
    removals = []
    remove = 1
    while remove == 1:
        row = int(input('Enter the row you want to remove an obstacle from '))
        col = int(input('Enter the column you want to remove an obstacle from '))        
        removals.append((row, col))
        remove = int(input('Enter 1 to continue removing obstacles. Otherwise, enter 0. '))
    return removals

def planner(start, goal, show = None):
    start.seen = True
    start.creach = 0
    start.cturn = 0
    start.cost = 0 + 0 + costtogoest(start, goal)
    start.parent = None
    start.orient = (-1,0) # assume robot starts facing the top of the grid
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
            print(goal.cost)
            break

        for neighbor in node.neighbors:
            if neighbor.done:
                continue

            creach = node.creach + node.cturn + 1
            cturn, angle = costtoturn(node, neighbor)

            if neighbor.seen:
                if (neighbor.creach + neighbor.cturn) <= (creach + cturn):
                    continue
                else:
                    onDeck.remove(neighbor)

            neighbor.seen = True
            neighbor.creach = creach
            neighbor.cturn = cturn
            neighbor.cost = creach + cturn + costtogoest(neighbor, goal)
            neighbor.parent = node
            neighbor.angle = angle
            neighbor.orient = (neighbor.row - node.row, neighbor.col - node.col)
            bisect.insort(onDeck, neighbor)

    path = [goal]
    while path[0].parent:
        path.insert(0, path[0].parent)
    return path

if __name__== "__main__":

    # GRID CREATION GOES HERE :D

    # DEFINE AND DRAW THE GRID DIMENSIONS AND THE OCCUPATION LOCATIONS OF THE GRID

    num_cols = int(input('Enter the number of columns in the grid '))
    num_rows = int(input('Enter the number of row in the grid '))

    # Set up the visual grid
    visual = VisualGrid(num_rows, num_cols)

    num_occupied = int(input('Enter the number of cells occupied by the robot '))

    robo_cells = []
    goal_cells = []

    for cell in range(0,num_occupied):
        print(f'For grid cell {cell+1} occupied by the robot')
        row_occ = int(input('Enter the row cell coordinate '))
        col_occ = int(input('Enter the column cell coordinate '))
        robo_cells.append((row_occ, col_occ))

    for cell in range(0, num_occupied):
        print(f'For grid square {cell + 1} occupied by the robot, enter the goal location')
        row_occ = int(input('Enter the goal row coordinate '))
        col_occ = int(input('Enter the goal column coordinate '))
        goal_cells.append((row_occ, col_occ))


    grid = [[' ' for _ in range(num_cols)] for _ in range(num_rows)]

    # UPPER, LOWER, SIDE WALLS

    #upper/lower walls
    for c in range(0, num_cols):
        grid[0][c] = '#'
        grid[num_rows-1][c] = '#'
        visual.color(0, c, BLACK)
        visual.color(num_rows-1, c, BLACK)

    #left/right walls
    for r in range(0, num_rows):
        grid[r][0] = '#'
        grid[r][num_cols-1] = '#'
        visual.color(r, 0, BLACK)
        visual.color(r, num_cols-1, BLACK)

    # print(grid)

    # ADD THE LOCATION OF THE ROBOT
    for cell in robo_cells:
        grid[cell[0]][cell[1]] = 'S'
        visual.write(cell[0], cell[1], 'S')

    # ADD THE GOAL LOCATION
    for cell in goal_cells:
        grid[cell[0]][cell[1]] = 'G'
        visual.write(cell[0], cell[1], 'G')

    # NEXT ADD THE OBSTACLES

    # Set an obstacle density variable
    obstacle_density = 0.5

    # Idea is to parse through the grid and to randomly assign obstacles 
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] != '#' and grid[row][col] != 'S' and grid[row][col] != 'G':
                if random.random() < obstacle_density:
                    grid[row][col] = '#'
                    visual.color(row, col, BLACK)

    visual.show(wait=0.005)

    ans = 5
    addition_list = []
    removal_list = []
    while ans != 0:
        ans = int(input('Enter 1 to add obstacles. \nEnter 2 to remove obstacles from the map. \nOtherwise, enter 0 if satisfied with the map: '))
        if ans == 1:
            addition_list = add_obstacles()
            for item in addition_list:
                grid[item[0]][item[1]] = '#'
                visual.color(item[0], item[1], BLACK)
                visual.show(wait=0.005)

        elif ans == 2:
            removal_list = remove_obstacles()
            print(f'Removal List: {removal_list}')
            for item in removal_list:
                grid[item[0]][item[1]] = ' '
                visual.color(item[0], item[1], WHITE)
                visual.show(wait=0.005)

    for row in grid:
        print(row)

    # create nodes based on free grid spaces
    nodes = []

    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] != '#':
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
    visual.show(wait="Hit return to start")

    def show(wait=0.005):
        for node in nodes:
            if node.done:   visual.color(node.row, node.col, TAN)
            elif node.seen: visual.color(node.row, node.col, GREEN)
            else:           visual.color(node.row, node.col, BLUE)
        visual.show(wait)

    path = planner(start, goal, show)

    for element in path:
        print(element, element.angle, element.cturn, element.creach, element.cost)

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