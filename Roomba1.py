

import cv2
import numpy as np
import random
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.animation as ani

random.seed(1)

# dimension the map is shrunk to
dim = 300
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
        # this is the position of the new node - actually just size of the heap + 1
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

def manhattan(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] + p2[1])

# takes in map, start location, stop location
# locations are tuples
# decides path using A*, returns path as array of tuples
# g will be linear, h will be euclidean distance
# can move to all 8 pixels/nodes surrounding current
# note that end and start are reversed. This is cuz the path starts looking from start
# so when it retraces its steps it ends at the start. Swapping these parameters fixes it
# (I could swap variables but i only just got it working)
def Astar(map, end, start):
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
                    # h = euclid([newX, newY], end)
                    # manhattan has less jagged turns
                    h = manhattan([newX, newY], end)
                    # make new node, add to queue
                    openQueue.push(AStarNode([newX, newY], current, g, h))
        closedQueue.append(current)

# takes in figure handle, center coordinates, radius, and colour, draws a circle
# for the asthetic(tm)
def draw(map, coords, radius, colour):
    for angle in range(0, 360, 5):
        x = int(round(radius * np.sin(np.radians(angle)) + coords[0]))
        y = int(round(radius * np.cos(np.radians(angle)) + coords[1]))
        if x >= 0 and y >= 0 and x < dim and y < dim:
            map[x][y] = colour
    return map

# takes in current velocity of roomba, obstacle information, plan, and timestep (index of plan we're on)
# startVelocity is maximum velocity, cannot go faster
# also takes in radius of roomba and obstacles
# checks that path is collision free based on distance and relative velocities
# if collision, calculates how much you must slow
def forbiddenVelocity(velocity, obstacles, plan, step, startVelocity, rRad, oRad):
    # deceleration from breaking in terms of pixels/timestep (also acceleration)
    breaking = 2
    n = len(obstacles)
    # initialize array for calculating relative velocities, distance, and time until collision
    # note - collision is defined as when the obstacle collides with the current path.
    # if no collision will occur, is inf
    # if obstacle is not moving (based on delay), is inf
    # current path is based on current trajectory - 2 linear lines intersecting
    # line based on current and future (5 steps?)
    # time is number of timesteps - once close to thresh, start slowing down
    relativeVelocities = np.zeros(n)
    # distance from roomba and obstacles to intersection
    rDists = np.zeros(n)
    oDists = np.zeros(n)
    times = np.zeros(n)
    # iterate through obstacles, calculate
    for i in range(n):
        # flag for if the distance between obstacle and roomba is less than combined radii
        close = False
        # check that obstacle is moving, and that it is still in play
        if obstacles[i].delay <= step and obstacles[i].x != np.inf:
            # for relative velocity, must take into account direction of roomba and obstacle
            # get roomba velocity based on current and next step

            temp = np.array(plan[step+1]) - np.array(plan[step])
            rNorm = np.sqrt(temp[0]**2 + temp[1]**2)
            # velocity shouldn't equal 0
            if velocity == 0:
                rDir = temp/rNorm # this is x,y vector
            else:
                rDir = velocity*temp/rNorm
            # get obstacles velocity
            # d is angle. Must convert
            oDir = [obstacles[i].v*np.cos(obstacles[i].d), obstacles[i].v*np.sin(obstacles[i].d)]
            # get relative velocity from reference frame of roomba
            relative = oDir-rDir
            # moving in negative direction actually makes it go up
            relative[1] = -1*relative[1]
            # convert to linear velocity
            # record so can compare
            relativeVelocities[i] = np.sqrt(relative[0]**2 + relative[1]**2)
            # based on distance of obstacle and relative velocities, find number of timesteps
            times[i] = relativeVelocities[i]/breaking
            # already past, no collision will occur, move on to next obstacle
            if times[i] <= 0:
                times[i] = np.inf
                dists[i] = np.inf
                continue
            # how far the point of intersection is
            # convert to standard from parametric form
            r = np.cross([plan[step][0], -plan[step][1], 1], [plan[step+1][0], -plan[step+1][1], 1])
            o = np.cross([obstacles[i].x, obstacles[i].y, 1], [obstacles[i].x + oDir[0], obstacles[i].y - oDir[1], 1])
            intersect = np.cross(r,o)
            z = intersect[2]
            intersect = intersect/z
            # will be (x, y, 1)
            # distance is from obstacle x y to intersect, taking into account radii of roomba and obstacles too
            oDists[i] = abs((np.sqrt((obstacles[i].x - intersect[0])**2 + (obstacles[i].y - intersect[1])**2)))
            rDists[i] = abs((np.sqrt((plan[step][0] - intersect[0])**2 + (plan[step][1] - intersect[1])**2)))
            rad = rRad + oRad
            # try to subtract the radius from the distances. Don't if they are already colliding
            if oDists[i] > rad:
                oDists[i] = oDists[i] - rad
            else:
                oDists[i] = 0 # collision imminent
            if rDists[i] > rad:
                rDists[i] = rDists[i] - rad
            else:
                rDists[i] = 0 # collision imminent
            # p=1
        else:
            relativeVelocities[i] = np.inf
            times[i] = np.inf
            oDists[i] = np.inf
            rDists[i] = np.inf

    # now based on relative velocities and distances, figure out time to collision
    # if time to collision is within 1 or 2 timesteps (so based on current velocity), start breaking
    # (return velocity reduced by breaking)
    # otherwise, return maximum speed up to default
    maxVelocity = velocity # start here, search through for max
    # if get through entire loop without any risk of collision, speed up
    slowFlag = 0
    #print(velocity)
    for i in range(n):
        if times[i] != np.inf:
            # this is the number of timesteps required for obstacle to reach
            oTime = oDists[i]/obstacles[i].v
            # and number of timesteps for roomba to reach at current velocity
            rTime = rDists[i]/velocity
            # check that its within 1 timestep
            # we have timesteps for obstacle to reach and timesteps to break. Must check timesteps for roomba to reach
            if abs(rTime-oTime) < relativeVelocities[i] or close is True:
                # only slow down when necessary -> obstacle reaches it with number of timesteps required to break
                if oTime <= times[i] + 1 or close is True:
                    # if maximum velocity is less than the slower velocity this would warrant, skip, otherwise replace
                    print("Slowing down for obstacle " + str(i))

                    # must constrain speed in negative direction
                    if velocity-breaking < -startVelocity:
                        print("oops!") # collision may occur, shouldn't happen
                        return -startVelocity
                    return (velocity-breaking)


    # got through loop without breaking, increase speed
    maxVelocity = maxVelocity + breaking
    if maxVelocity >= startVelocity:
        maxVelocity = startVelocity
        return maxVelocity
    print("Speeding up")
    return maxVelocity


# takes in the map, the A* plan+angle, roomba velocity, and a list of all the obstacles (for simulation purposes)
# moves roomba and obstacles, observes field and may change velocity to slow down according to forbidden map
# if moving/unexpected obstacles, applies modification to path
# obstacles have x, y, v, d (coordinates, lin velocity, direction in degrees)
def move(originalMap, plan, velocity, obstacles):
    # start velocity saved as maximum
    startVelocity = velocity
    flag = True # gets switched to False once end point reached
    # find how many steps are in the path
    num = len(plan)
    # plot the path
    for i in range(num):
        # grey out path
        originalMap[plan[i][0]][plan[i][1]] = 0.5

    originalMap = np.multiply(originalMap, 255)
    # hope theres no aliasing
    map = originalMap # convert to numpy array

    fig = plt.figure(1)
    moviewriter = ani.PillowWriter()
    x = plan[0][0] # start position of the roomba
    y = plan[0][1]
    # roomba colour, can go from 0 to 255
    rColour = 200
    # obstacle color
    oColour = 200
    # radius of obstacles and roomba
    oRad = 10
    rRad = 20
    numObstacles = len(obstacles)
    count = 0 # where in path the roomba is
    timer = 0 # counts for when the obstacles on delay start moving
    with moviewriter.saving(fig, 'Roomba.gif', dpi=100):
        while flag:
            # using the velocities and direction of the moving obstacles, adjusts velocity to avoid collision on path
            # ignores obstacles since path avoids them and stationary obstacles don't exist for moving obstacles
            velocity = forbiddenVelocity(velocity, obstacles, plan, count, startVelocity, rRad, oRad)

            # redraw (and recreate map) map
            map = np.array(originalMap)
            # draw each step
            map[x][y] = rColour
            for i in range(numObstacles):
                # only move if number of timesteps is greater than delay
                # prevents far away obstacles from moving before the roomba gets close
                # helps the roomba encounter more obstacles so that you can acutally see the forbidden velocity thing
                if obstacles[i].delay <= timer:
                    xo = obstacles[i].x
                    yo = obstacles[i].y

                    # if xo is 0 then the obstacle has left the map - skip this obstacle
                    if xo == np.inf:
                        continue
                    map[xo][yo] = oColour
                    vo = obstacles[i].v
                    do = obstacles[i].d

                    # draw circle around obstacle center, only draw if obstacle moving
                    map = draw(map, [obstacles[i].x, obstacles[i].y], oRad, oColour)

                    # change x and y of the obstacles based on v and d (v and d will always be the same)
                    # because I'm doing a grid, simplify by rounding.
                    newX = int(np.round(xo + vo*np.cos(do)))
                    newY = int(np.round(yo + vo*np.sin(do)))

                    if newX < 0 or newX >= dim or newY < 0 or newY >= dim:
                        obstacles[i].x = np.inf # just set x, no need to also set y
                    else:
                        obstacles[i].x = newX
                        obstacles[i].y = newY
            timer += 1
            # assess obstacles and self, calculate forbidden velocity map, if necessary, change velocity
            # assume velocity is constant for now
            # velocity is how many steps of the path it skips
            count += velocity
            # if chased off the map, stop and wait I guess. Start spot is 'safe'
            if count < 0:
                count = 0
            # if exceeds, just go to end point
            if count >= num-1:
                x = plan[num-1][0]
                y = plan[num-1][1]
                map = draw(map, plan[num-1], rRad, rColour)
                flag = False
            else:
                x = plan[count][0]
                y = plan[count][1]
                map = draw(map, plan[count], rRad, rColour)
            # send map into drawing function to be drawn correctly

            plt.imshow(map, vmin=0, vmax=255)
            moviewriter.grab_frame()
            # clear frame to redraw
            plt.clf()
        moviewriter.finish()


# angle is dependent on the direction between 1 point and another
def getAngle(plan):
    num = len(plan) - 1
    for i in range(num):
        dx = plan[i][0] - plan[i+1][0]
        dy = plan[i][1] - plan[i+1][1]
        # check if dx is 0 -> 90 degrees
        if dx == 0:
            angle = np.pi/2
        elif dy == 0:
            angle = 0
        else:
            # find the angle in radians
            angle = np.arctan(dy/dx)
        plan[i].append(angle) # add angle to list
    return plan


class Obstacle:
    # everything is randomly initialized as default, though you can pass it hard values
    def __init__(self, plan, x=None, y=None, v=None, d=None, delay=None):
        # x and y are current coordinates
        if x is None:
            x = random.randint(0, dim-1)
        if y is None:
            y = random.randint(0, dim-1)
        if v is None:
            v = random.randint(1,8)
        if d is None:
            d = random.random()*2*np.pi
        if delay is None:
            delay = random.randint(0, len(plan))
        self.x = x
        self.y = y
        # v is linear velocity
        self.v = v
        # direction is angle in rad wrt global coords
        self.d = d
        # number of timesteps before obstacle starts moving
        self.delay = delay

def main():
    image = "OccupancyMapDrawn.png"
    numObstacles = 50 # a number of these likely won't be in the way
    # simulation time will be for as long as the roomba requires to reach goal. Velocity is pixels/timestep
    # so velocity of 1 does every single step. Velocity of 2 does every other etc.
    velocity = 5

    # converts image to 0's and 1's
    # 0 is empty, 1 is blocked
    map = makeMap(image)
    # !!!!!!!! get these in a better way later - interactive clicking. can do in makeMap?
    # also, make sure they are in an array, not a tuple!!!
    start = [5,5]
    end = [245,245]

    # this plan is just x,y coordinates of A* path
    plan = Astar(map, start, end)
    # plan becomes (x,y,th), as we need the angle of the roomba too for drawing purposes
    #plan = getAngle(plan)

    # minimum colliding example
    # obstacles = [Obstacle(plan, 100, 20, 4, 2, 1)]

    # get obstacles - random start point initialization, random constant velocities (including directions)
    # each row is an obstacle, consisting of x, y, linear velocity, angle (in degrees), delay in timesteps before moving
    obstacles = [Obstacle(plan) for _ in range(numObstacles)]
    # send map and plan into move, along with roomba velocity and obstacles
    move(map, plan, velocity, obstacles)

main()

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