# CREATING A NEW USER-DEFINED GRID BUT WITH A DIFFERENT GRID INITIALIZATION

from visualgrid import VisualGrid
import math
import random

WHITE = [1.000, 1.000, 1.000]
GREEN = [0.816, 0.914, 0.753]
BLUE = [0.745, 0.867, 0.945]
PINK = [0.965, 0.757, 0.698]
TAN = [0.914, 0.788, 0.667]
ROSE = [0.965, 0.722, 0.816]
GOLD = [0.973, 0.773, 0.486]
RED = [0.929, 0.525, 0.596]
BLACK = [0.000, 0.000, 0.000]

# nodes = []

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

visual.show(wait=0.005)

for row in grid:
    print(row)

input('Hit return to end')