#!/usr/bin/env python3
#
#   astar.py
#
#   This provides the A* algorithm:
#
#      from astar import AStarNode, astar
#
#      path = astar([nodes], startnode, goalnode)
#
#   where the nodes should inherit from AStarNode and define costs to
#   connect as well as estimate costs to go.
#
#      class Node(AStarNode):
#          def __init__(self):
#              super().__init__()
#
#          def costToConnect(self, other):
#              return some_distance_metric
#
#          def costToGoEst(self, other):
#              return some_distance_metric
#
#   and the nodes neighbors should have been connected (to form a graph):
#
#      node.neighbors = [list_of_neighbors]
#
import bisect
import math


#
#   Basic A* Search Tree Node Class
#
#   Build on this class to define your nodes.
#
class AStarNode:
    def __init__(self):
        # Edges = set of neighbors.  You need to fill in.
        self.neighbors = set()

        # Reset the A* search tree information
        self.reset()

    def reset(self):
        # Clear the status, connection, and costs for the A* search tree.
        #   TRUNK:  done = True
        #   LEAF:   done = False, seen = True
        #   AIR:    done = False, seen = False
        self.done     = False
        self.seen     = False
        self.parent   = None
        self.creach   = 0               # Known/actual cost to get here
        self.ctogoest = math.inf        # Estimated cost to go from here

    # Actual travel and estimated to-go costs.
    def costToConnect(self, other):
        print("Please add costToConnect(self, other) to your Node definition!")
        raise NotImplementedError

    def costToGoEst(self, other):
        print("Please add costToGoEst(self, other) to your Node definition!")
        raise NotImplementedError
    
    # Define the "less-than" to enable sorting in A*. Use total cost estimate.
    def __lt__(self, other):
        return (self.creach + self.ctogoest) < (other.creach + other.ctogoest)


#
#   A* Planning Algorithm
#
def astar(nodes, start, goal):
    # Clear the A* search tree information.
    for node in nodes:
        node.reset()

    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    # Begin with the start node on-deck.
    start.done     = False
    start.seen     = True
    start.parent   = None
    start.creach   = 0
    start.ctogoest = start.costToGoEst(goal)
    bisect.insort(onDeck, start)

    # Continually expand/build the search tree.
    while True:
        # Make sure we have something pending in the on-deck queue.
        # Otherwise we were unable to find a path!
        if not (len(onDeck) > 0):
            return None

        # Grab the next node (first on deck).
        node = onDeck.pop(0)

        # Mark this node as done and check if the goal is thereby done.
        node.done = True
        if goal.done:
            break

        # Add the neighbors to the on-deck queue (or update)
        for neighbor in node.neighbors:
            # Skip if already done.
            if neighbor.done:
                continue

            # Compute the cost to reach the neighbor via this new path.
            creach = node.creach + node.costToConnect(neighbor)

            # Just add to on-deck if not yet seen (in correct order).
            if not neighbor.seen:
                neighbor.seen     = True
                neighbor.parent   = node
                neighbor.creach   = creach
                neighbor.ctogoest = neighbor.costToGoEst(goal)
                bisect.insort(onDeck, neighbor)
                continue

            # Skip if the previous path to reach (cost) was same or better!
            if neighbor.creach <= creach:
                continue

            # Update the neighbor's connection and resort the on-deck queue.
            # Note the cost-to-go estimate does not change.
            neighbor.parent = node
            neighbor.creach = creach
            onDeck.remove(neighbor)
            bisect.insort(onDeck, neighbor)

    # Build the path.
    path = [goal]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Return the path.
    return path
