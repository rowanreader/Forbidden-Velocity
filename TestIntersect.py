import numpy as np

roomba = [1,0]
obstacle = [5,5]

rDir = [0,1]
oDir = [-1,0]

r = np.cross([roomba[0], roomba[1], 1],[roomba[0] + rDir[0], roomba[1] + rDir[1],1])

o = np.cross([obstacle[0], obstacle[1],1],[obstacle[0] + oDir[0], obstacle[1] + oDir[1],1])

intersect = np.cross(r,o)
print(intersect)
            
