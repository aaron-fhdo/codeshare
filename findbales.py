#!/usr/bin/env python
# coding: utf-8


import time
import matplotlib.pyplot as plt
from threading import Thread
import numpy as np


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares

            # Get node position
            node_position = (
                current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:  # obstacle
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = abs(child.position[0] - end_node.position[0]) + abs(
                child.position[1] - end_node.position[1]) + 1  # heuristic to manhattan
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


dbg = 0

startSBV = [0, 0]
startMBV = [1, 0]
startLBV = [2, 0]
startWBV = [4, 0]
depositSBV = [2, 0]
depositMBV = [3, 0]
depositLBV = [4, 0]
depositWBV = [5, 0]


# In[23]:


# description: function to generate bale/object locations
# params: arena matrix and display flag
# return: small/medium/large bale/obstacle locations

def generate_arena(arena, disp):
    small = np.where(arena == 1)
    small_x = np.asarray(small[0]).flatten()
    small_y = np.asarray(small[1]).flatten()
    med = np.where(arena == 2)
    med_x = np.asarray(med[0]).flatten()
    med_y = np.asarray(med[1]).flatten()
    large = np.where(arena == 3)
    large_x = np.asarray(large[0]).flatten()
    large_y = np.asarray(large[1]).flatten()
    weed = np.where(arena == 4)
    weed_x = np.asarray(weed[0]).flatten()
    weed_y = np.asarray(weed[1]).flatten()
    obs = np.where(arena == 5)
    obs_x = np.asarray(obs[0]).flatten()
    obs_y = np.asarray(obs[1]).flatten()

    if(disp):  # only if disp is true
        plt.plot(small_y, small_x, 'rs')
        plt.plot(startSBV[1], startSBV[0], 'r^')
        plt.plot(med_y, med_x, 'bs')
        plt.plot(startMBV[1], startMBV[0], 'b^')
        plt.plot(large_y, large_x, 'gs')
        plt.plot(startLBV[1], startLBV[0], 'g^')
        plt.plot(obs_y, obs_x, 'ko')
        plt.plot(weed_y, weed_x, 'kx')
        plt.axis([-1, 16, -1, 8])
        plt.gca().invert_yaxis()
        plt.grid(color='k', linestyle='-', linewidth=.2)
        plt.title("Arena")
        plt.show()
    return small_x, small_y, med_x, med_y, large_x, large_y, obs_x, obs_y, weed_x, weed_y


# description: function to generate field mask to treat all other objects as obstacles
# params: x - arena matrix and unmask_val - value to be unmasked
# return: masked arena

def mask_arena(x, unmask_val):
    new_arena = np.zeros([8, 16], 'int')
    for i in range(len(x)-1):
        for j in range(len(x[i])-1):
            if (x[i, j] == 5):
                new_arena[i, j] = 1
            '''
                if(unmask_val==1):
                    if ((x[i,j]==2)|(x[i,j]==3)|(x[i,j]==5)):
                        new_arena[i,j]=1
                elif(unmask_val==2):
                    if ((x[i,j]==1)|(x[i,j]==3)|(x[i,j]==5)):
                        new_arena[i,j]=1
                elif(unmask_val==3):
                    if ((x[i,j]==1)|(x[i,j]==2)|(x[i,j]==5)):
                        new_arena[i,j]=1
                '''

    return new_arena

    # print(rand_i,rand_j)


def insert_checkpoints(list_x, list_y, start, deposit, val):
    if(val == 0):
        list_x = np.insert(list_x, 0, start[0])
        list_y = np.insert(list_y, 0, start[1])
        list_x = np.append(list_x, deposit[0])
        list_y = np.append(list_y, deposit[1])
    else:
        length = len(list_x)
        index = np.arange(0, length, val)
        list_x = np.insert(list_x, 0, start[0])
        list_y = np.insert(list_y, 0, start[1])
        list_x = np.insert(list_x, index[1:], deposit[0])
        list_y = np.insert(list_y, index[1:], deposit[1])
        list_x = np.append(list_x, deposit[0])
        list_y = np.append(list_y, deposit[1])
    return list_x, list_y


def smallbalecollector():
    if(dbg == 1):
        print("Bale coordinates:", small_x, small_y, "\n")
    print("Starting SBV..\n")
    global pathS
    pathS = []
    ctrS = 0
    for i in range(len(small_x)-1):
        pathS = pathS + \
            astar(maskS, (small_x[i], small_y[i]),
                  (small_x[i+1], small_y[i+1]))
        if(dbg == 1):
            print(pathS, "\n")
    print("Finishing SBV..\n")
    return pathS


def medbalecollector():
    if(dbg == 1):
        print("Bale coordinates:", med_x, med_y, "\n")
    print("Starting MBV..\n")
    global pathM
    pathM = []
    for i in range(len(med_x)-1):
        pathM = pathM + \
            astar(maskM, (med_x[i], med_y[i]), (med_x[i+1], med_y[i+1]))
        if(dbg == 1):
            print(pathM, "\n")
    print("Finishing MBV..\n")
    return pathM


def largebalecollector():
    if(dbg == 1):
        print("Bale coordinates:", large_x, large_y, "\n")
    print("Starting LBV..\n")
    global pathL
    pathL = []
    for i in range(len(large_x)-1):
        pathL = pathL + \
            astar(maskL, (large_x[i], large_y[i]),
                  (large_x[i+1], large_y[i+1]))
        if(dbg == 1):
            print(pathL, "\n")
    print("Finishing LBV..\n")
    return pathL


def weedbalecollector():
    if(dbg == 1):
        print("Bale coordinates:", large_x, large_y, "\n")
    print("Starting WBV..\n")
    global pathW
    pathW = []
    for i in range(len(weed_x)-1):
        pathW = pathW + \
            astar(maskW, (weed_x[i], weed_y[i]), (weed_x[i+1], weed_y[i+1]))
        if(dbg == 1):
            print(pathL, "\n")
    print("Finishing WBV..\n")
    return pathW


# #### Arena Loading and Visualization

arena = np.loadtxt("testField_BruteForce.txt", 'int')

padding = []
padding = np.zeros([8, 16], 'int')
print(padding)
padding[1:7, 1:15] = arena
print(padding)
arena = padding
small_x, small_y, med_x, med_y, large_x, large_y, obs_x, obs_y, weed_x, weed_y = generate_arena(
    arena, disp='true')



small_x, small_y = insert_checkpoints(
    small_x, small_y, startSBV, depositSBV, 3)
med_x, med_y = insert_checkpoints(med_x, med_y, startMBV, depositMBV, 2)
large_x, large_y = insert_checkpoints(
    large_x, large_y, startLBV, depositLBV, 1)
weed_x, weed_y = insert_checkpoints(weed_x, weed_y, startWBV, depositWBV, 0)

print(small_x, small_y)
print(med_x, med_y)
print(large_x, large_y)


maskS = mask_arena(arena, 1)
maskM = mask_arena(arena, 2)
maskL = mask_arena(arena, 3)
maskW = mask_arena(arena, 4)


# ##### Path Finding in Concurrent Threads
t1 = Thread(target=smallbalecollector)
t2 = Thread(target=medbalecollector)
t3 = Thread(target=largebalecollector)

dbg = 0

starttime = time.time()
t1.start()
t2.start()
t3.start()

pathS = t1.join()
pathM = t2.join()
pathL = t3.join()
print("\n\nExec time is ", time.time()-starttime)

dbg = 0



# #### Path Visualiation

def visualize_path(path, arena):

    x_val = [x[0] for x in path]
    y_val = [x[1] for x in path]

    # plt.plot(startSBV,'rx')
    # plt.plot(depositSBV,'go')

    plt.plot(small_y[1:-1], small_x[1:-1], 'rs')
    plt.plot(med_y[1:-1], med_x[1:-1], 'bs')
    plt.plot(large_y[1:-1], large_x[1:-1], 'gs')
    plt.plot(weed_y[1:-1], weed_x[1:-1], 'kx')
    plt.plot(obs_y, obs_x, 'ko')
    plt.axis([-1, 16, -1, 8])
    plt.gca().invert_yaxis()
    plt.grid(color='k', linestyle='-', linewidth=.2)

    plt.plot(y_val, x_val)
    plt.show()


visualize_path(pathS, arena)
visualize_path(pathM, arena)
visualize_path(pathL, arena)
visualize_path(pathW, arena)
