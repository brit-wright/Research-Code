'''visualgrid.py

   Leverage Matplotlib to create a visual representation of a 2D grid.
   To use:

     v = VisualGrid(rows, cols)    Create the figure

     v.color(row, col, color)      Set the RGB color at (row,col)
     v.write(row, col, text)       Set the text at (row,col)
     v.show()                      Draw/flash the figure
     v.show(wait=True)             Show the figure and wait for return
     v.show(wait="Message")        Show the figure and wait with message
     v.show(wait=0.1)              Show the figure and wait 0.1seconds

     where row, col are integers.  color = [r, g, b] is a list of
     three floating point numbers between 0 and 1.
'''

import numpy as np
import matplotlib.pyplot as plt


#
#   Grid Visualization Object
#
#   Create a figure for an (rows) x (column) grid.  The X-axis will be
#   the columns (to the right) and the Y-axis will be the rows (top
#   downward).
#
class VisualGrid:
    # Initialization.
    def __init__(self, rows, cols):
        # Save the dimensions.
        assert int(rows)>0, "Must specify a positive number of rows"
        assert int(cols)>0, "Must specify a positive number of cols"
        self.rows = int(rows)
        self.cols = int(cols)

        # Create/activate a figure, clear, set up axes, turn off labels.
        plt.figure()
        plt.clf()
        self.ax = plt.axes()
        self.ax.axis('off')

        # Create the color boxes, called "quilt", defaulting to white.
        self.colors = np.ones((self.rows,self.cols,3))
        self.quilt  = self.ax.imshow(
            self.colors, interpolation='none', aspect='equal',
            extent=[0, self.cols, 0, self.rows], zorder=0)
        
        # Create the labels (defaulting to space).
        self.labels = [[self.label(r, c, ' ')
                        for c in range(self.cols)] for r in range(self.rows)]

        # Draw the grid lines, zorder 1 means draw after zorder 0 elements.
        for row in range(self.rows+1):
            self.ax.axhline(row, lw=1, color='b', zorder=1)
        for col in range(self.cols+1):
            self.ax.axvline(col, lw=1, color='b', zorder=1)

        # Add the external labels.
        for row in range(self.rows):
            self.label(row,        -1, str(row))
            self.label(row, self.cols, str(row))
        for col in range(self.cols):
            self.label(       -1, col, str(col))
            self.label(self.rows, col, str(col))

        # Show the grid.
        self.show()

    # Create a text label in a grid element.
    def label(self, row, col, text):
        return plt.text(0.33 + col, self.rows - 0.67 - row, text)


    # Change the color at (row,col).
    def color(self, row, col, color):
        assert int(row) >= 0 and int(row) < self.rows, "row out of range"
        assert int(col) >= 0 and int(col) < self.cols, "col out of range"
        self.colors[int(row)][int(col)] = color

    # Update the text at (row,col).
    def write(self, row, col, text):
        assert int(row) >= 0 and int(row) < self.rows, "row out of range"
        assert int(col) >= 0 and int(col) < self.cols, "col out of range"
        self.labels[int(row)][int(col)].set_text(text)

    # Show the visualization.
    def show(self, wait=False):
        # Update the quilt (color boxes).
        self.quilt.set(array = self.colors)
        # Force the figure to update.  And wait to hit enter.
        plt.show(block=False)
        if isinstance(wait, float):
            plt.pause(max(0.001, wait))
        elif wait:
            plt.pause(0.001)
            input('Hit return to continue' if wait is True else wait)
        else:
            plt.pause(0.001)
