from visualgrid import VisualGrid
import math

BLACK = [0.000, 0.000, 0.000]

# Create the back-end of the grid
num_cols = int(input('Enter the number of columns in the grid '))
num_rows = int(input('Enter the number of row in the grid '))

num_occupied = int(input('Enter the number of cells occupied by the robot '))

robo_cells = []
robo_rows = []
robo_cols = []

for cell in range(0,num_occupied):
    print(f'For grid cell {cell+1} occupied by the robot')
    row_occ = int(input('Enter the row cell coordinate '))
    col_occ = int(input('Enter the column cell coordinate '))
    robo_cells.append((row_occ-1, col_occ-1))
    robo_rows.append(row_occ-1)
    robo_cols.append(col_occ-1)
    

obstacle_choice = int(input('Enter 1 to manually add obstacle locations and 2 to have obstacles randomly generated and 3 to use a default obstacle map'))

# If it's manual then I can keep using the normal grid setup
# If it's random I could also keep using the normal grid setup
# If I choose a default obstacle map then I'll probs have to convert to a string first

# Create the grid as a 2-dimensional list
grid = []

# Doing a row by row construction of the grid without obstacles/robot included
spaces = num_cols - 2

for row in range(num_rows):

    # UPPER AND LOWER WALLS
    if row == 0 or row == num_rows-1:
        grid.append('#'*num_cols)
    
    elif row in robo_rows:
        
        # get the row,col pairs that are occupied by the bot
        interest = [(r,c) for (r,c) in robo_cells if r == row]

        # Now that I have the coordinates of interest I can start by defining some
        # formula that allows me to place the # where they should go
        # I could lowkey just do list to string lol

        row_list = []
        col_interest = [0, num_cols-1]
        
        for tup in interest:
            col_interest.append(tup[1])

        # from here I could create a list to just append the elements to
        for iter in range(num_cols):
            if iter in col_interest:
                row_list.append('#')
            else:
                row_list.append(' ')

        # then convert list to string
        row_str = ''.join(row_list)

        # then append that string to the grid
        grid.append(row_str)

    # WALLS WITHOUT ROBOT OR OBSTACLES
    else:
        grid.append('#' + ' '*spaces + '#')
print(grid)


# grid = [[' ' for _ in range(cols)] for _ in range(rows)]

# Probably print disclaimer about the boundaries?
# print('By default this code will consider the outer wall, which defines the rectangular boundary of the grid to be' \
# 'impassable (an obstacle), meaning that the start and goal nodes for the robot will never be outside the grid')

# response = 1
# response = int(input('To proceed like this press 1. Otherwise, press 0 '))

# not_wall = []
# ans = 0
# if response == 0:
#     while ans!=1:
#         n_wall_row = int(input('Enter the row of the non-wall boundary '))
#         n_wall_col = int(input('Enter the column of the non-wall boundary '))
#         not_wall.append((n_wall_row, n_wall_col))
#         ans = int(input('Press enter to continue and 1 if there are no more non-wall boundaries to include '))

# Draw the grid
