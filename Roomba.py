import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# dimension the map is shrunk to
dim = 15
# converts image into map, returns map as 2D array
# _ = empty space, X = known obstacle
def makeMap(file):
    map = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # reduce size so easier to compute
    map = cv2.resize(map, (dim, dim))
    #cv2.imshow("img", map)
    #cv2.waitKey(0)
    # map = np.array(map)
    # map[map > 100] = 0
    # map[map < 100] = 1

    size = map.shape
    row = size[0]
    col = size[1]
    newMap = [[0 for j in range(col)] for i in range(row)]
    # for each row
    for i in range(row):
        for j in range(col):
            if map[i][j] < 100:
                newMap[i][j] = 1

    return newMap

# takes in 2D array, and path, modifies map to include path
# prints out line by line
def displayArray(map, path):
    size = len(map)
    # start and end nodes will have special, separate characters
    length = len(path) - 1
    # path is given reversed, so do end node first
    x = path[0][0]
    y = path[0][1]
    map[x][y] = 'E'
    for i in range(1,length):
        x = path[i][0]
        y = path[i][1]
        map[x][y] = 'X'
    x = path[i+1][0]
    y = path[i+1][1]
    map[x][y] = 'S'
    # should be square
    for i in range(size):
        print(map[i])

# makes a heap
class Heap():
    def __init__(self, maxSize):
        self.maxSize = maxSize + 1 # add 1 cuz 1st index is taken
        # first element contains size, the rest are Astar nodes set to 0 for everything
        self.heap = [0] + [AStarNode()]*maxSize
        # need a way to keep track of duplicates
        # each hash[x][y] contains the index in the hash table. Default is 0
        # based on idea of hash, however shouldn't get collisions between non-duplicates
        self.hash = [[0 for i in range(dim)] for j in range(dim)]

    # takes heap and position of suspect node, fixes heap
    # assumes issue is that current node is larger than children, not that current node is smaller than parents
    # shouldn't be an issue unless something really screwy happens
    def heapify(self, pos):
        # check if heap is empty
        if self.heap[0] == 0:
            return
        # check if leaf because recursive function
        # is a leaf if it is in the latter half of the array (and less than the actual size) *0 is length
        if self.heap[0]//2 < pos <= self.heap[0]:
            return
        # check if current is larger than children
        if self.heap[pos].F > self.heap[2*pos].F or self.heap[pos].F > self.heap[(2*pos)+1].F:
            # swap with min child
            temp = self.heap[pos]
            if self.heap[2*pos].F < self.heap[(2*pos) + 1].F:
                # gotta swap the hash first
                x1 = self.heap[pos].x
                y1 = self.heap[pos].y
                x2 = self.heap[2 * pos].x
                y2 = self.heap[2 * pos].y
                self.hash[x1][y1] = 2*pos
                self.hash[x2][y2] = pos

                self.heap[pos] = self.heap[2*pos]
                self.heap[2*pos] = temp
                self.heapify(2*pos)

            else:
                x1 = self.heap[pos].x
                y1 = self.heap[pos].y
                x2 = self.heap[(2*pos) + 1].x
                y2 = self.heap[(2*pos) + 2].y
                self.hash[x1][y1] = (2*pos) + 1
                self.hash[x2][y2] = pos

                self.heap[pos] = self.heap[(2*pos) + 1]
                self.heap[(2*pos) + 1] = temp
                self.heapify((2*pos) + 1)




    # push item onto heap
    def push(self, node):
        # this is the position of the new node - acutally just size of the heap + 1
        current = self.heap[0] + 1

        x = node.x
        y = node.y
        # check if already expanded (in which case that was cheaper, no need to explore)
        if self.hash[x][y] == -1:
            return
        # check if already in the heap
        elif self.hash[x][y] != 0:
            # if new one is cheaper, replace and return, otherwise do nothing and return
            if self.heap[self.hash[x][y]].F > node.F:
                self.heap[self.hash[x][y]] = node
            return

        # set node in last place
        self.heap[current] = node

        # gotta add to make sure length isn't wrong
        self.heap[0] += 1
        # parent is halfway between current location and start of heap (integer division)
        parent = current//2
        # if parent index is 0, its the size value
        if parent == 0:
            self.hash[x][y] = current
            return

        # f is the g+h of the node
        while self.heap[current].F < self.heap[parent].F:
            # swap hash positions
            px = self.heap[parent].x
            py = self.heap[parent].y
            # this is the current, will have to change x and y later. swap for parent's index
            self.hash[x][y] = parent
            self.hash[px][py] = current

            # swap the current and parent nodes
            temp = self.heap[parent]
            self.heap[parent] = self.heap[current]
            self.heap[current] = temp
            # now current is at parent's index, and the new parent must be found
            current = parent
            # change x and y to parents
            x = px
            y = py
            parent = current//2

            # current has moved to the top of the heap
            if parent == 0:
                return

    # removes and returns the minimum element on the heap (this is the top)
    def pop(self):
        first = self.heap[1]
        # check if heap is empty
        if self.heap[0] == 0:
            return first

        # replace minimum element with maximum element
        self.heap[1] = self.heap[self.heap[0]]
        # set coordinates in hash as well
        x = self.heap[1].x
        y = self.heap[1].y
        oldX = first.x
        oldY = first.y
        # is now at position 1 on heap
        self.hash[x][y] = 1
        # is no longer something we care about - anything that explores this will be more expensive
        self.hash[oldX][oldY] = -1

        # remove 1 from stored length
        self.heap[0] -= 1
        # heapify to correct, should go all the way down, index in question is index 1
        # (index 0 is size)
        self.heapify(1)
        return first


class AStarNode():
    # initialize node with coordinates(x,y), cost(G), estimate(H), and parent
    # default g is infinity for min heap purposes
    def __init__(self, coord=[0,0], parent=None, g=np.inf, h=0):
        self.x = coord[0]
        self.y = coord[1]
        self.parent = parent
        self.G = g
        self.H = h
        self.F = g + h # just for convenience's sake when doing min heap

# calculates euclidian distance between 2 points (given as tuples)
# this is the heuristic method for A*
def euclid(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# takes in map, start location, stop location
# locations are tuples
# decides path using A*, returns path as array of tuples
# g will be linear, h will be euclidean distance
# can move to all 8 pixels/nodes surrounding current
def Astar(map, start, end):
    # initialize start node
    startNode = AStarNode(start, None, 0, 0)
    # initialize queues, open one must be ordered so make it a min heap
    size = dim**2 # the number should be number of pixels

    openQueue = Heap(size)
    openQueue.push(startNode)
    closedQueue = []
    # openQueue is unexplored nodes that you can explore (neighbours)

    while openQueue.heap:
        # openQueue is a min heap, pop off 1st element (cheapest node)
        current = openQueue.pop()
        oldX = current.x
        oldY = current.y

        # check if goal reached
        if [oldX,oldY] == end:
            # closed Queue is all explored nodes. last one is goal, can follow parents to find path
            path = [[oldX, oldY]]
            temp = end
            # trace back from goal by following parent until you reach the start
            while temp != start:
                # append parent coordinates
                path.append([current.parent.x, current.parent.y])
                # move current up to parent
                current = current.parent
                # want to be mutable, so array, not tuple
                temp = [current.x, current.y]
            return path

        # generate all successors (8 of them) as nodes
        # if a coordinate is negative, set g to inf
        for i in range(-1,2):
            for j in range(-1,2):
                newX = oldX + i
                newY = oldY + j
                # skip self node, make sure within range, skip obstacle points
                if (i != 0 or j != 0) and (0 <= newX < dim and 0 <= newY < dim) and map[newX][newY] == 0:

                    g = current.G + 1 # assume each step is same
                    # find euclidian distance between prospective node and end goal
                    h = euclid([newX, newY], end)
                    # make new node, add to queue
                    openQueue.push(AStarNode([newX, newY], current, g, h))
        closedQueue.append(current)


# takes in the map, the A* plan+angle, roomba velocity, and a list of all the obstacles (for simulation purposes)
# moves roomba and obstacles, observes field and may change velocity to slow down according to forbidden map
# if moving/unexpected obstacles, applies modification to path
# obstacles have x, y, v, d (coordinates, lin velocity, direction in degrees)
def move(map, plan, velocity, obstacles):
    img = plt.imshow(map)
    plt.plot(map[0],map[1])
    while True:
        # draw each step

        # assess obstacles and self, calculate forbidden velocity map, if necessary, change velocity

        # move 1 time step (1 sec?)
        # if exceeds plan, set to final goal (can have obstacles still move tho), draw last time, return
        pass


# angle is dependent on the direction between 1 point and another
def getAngle(plan):
    num = len(plan) - 1
    for i in range(num):
        dx = plan[i][0] - plan[i+1][0]
        dy = plan[i][1] - plan[i+1][1]
        # find the 4 quadrant angle, convert to degrees
        angle = np.rad2deg(np.arctan(dy, dx))
        plan[i].append(angle) # add angle to list
    return plan


class Obstacle:
    # everything is randomly initialized as defalt, though you can pass it hard values
    def __init__(self, x=random.randint(0,dim), y=random.randint(0,dim), v=random.randint(0,20), d=random.randint(0,360)):
        # x and y are current coordinates
        self.x = x
        self.y = y
        # v is linear velocity
        self.v = v
        # direction is angle in degrees wrt global coords
        self.d = d


def main():
    image = "OccupancyMapDrawn.png"
    numObstacles = 10 # a number of these likely won't be in the way
    # simulation time will be for as long as the roomba requires to reach goal. Velocity is pixels/timestep
    # so velocity of 1 does every single step. Velocity of 2 does every other etc.
    velocity = 10

    # converts image to 0's and 1's
    # 0 is empty, 1 is blocked
    map = makeMap(image)
    # !!!!!!!! get these in a better way later - interactive clicking. can do in makeMap?
    # also, make sure they are in an array, not a tuple!!!
    start = [10,10]
    end = [25,25]

    # this plan is just x,y coordinates of A* path
    plan = Astar(map, start, end)
    print("Here2!")
    # plan becomes (x,y,th), as we need the angle of the roomba too
    plan = getAngle(plan)
    # get obstacles - random start point initialization, random constant velocities (including directions)
    # each row is an obstacle, consisting of x, y, linear velocity, angle (in degrees)
    obstacles = [Obstacle() for i in range(numObstacles)]

    # send map and plan into move, along with roomba velocity and obstacles
    move(map, plan, velocity, obstacles)

#main()

# test heap works
# run with smaller map maybe?
# c = [1, 1]
# p = None
# g = 0
# a1 = AStarNode(c,p,g,4)
# a2 = AStarNode(c,p,g,32)
# a3 = AStarNode(c,p,g,7)
# a4 = AStarNode(c,p,g,65)
# a5 = AStarNode(c,p,g,17)
# a6 = AStarNode(c,p,g,10)
# a7 = AStarNode(c,p,g,23)
#
# h = Heap(10)
# h.push(a1)
# h.push(a2)
# h.push(a3)
# h.push(a4)
# h.push(a5)
# h.push(a6)
# h.push(a7)
#
# array = np.zeros(h.heap[0])
# for i in range(1, h.heap[0]+1):
#     array[i-1] = h.heap[i].H
#
# print(array)
# print(h.heap[0])

# test AStar works
# Note: this was tested for 15x15
# map = [[0 for i in range(dim)] for j in range(dim)]
# start = [0,1]
# end = [14,8]
# # put a few obstacles in
# map[2][2] = 1
# map[3][3] = 1
# map[1][3] = 1
# map[2][3] = 1
# map[3][3] = 1
# map[4][3] = 1
#
# path = Astar(map, start, end)
# print(path)
# displayArray(map, path)