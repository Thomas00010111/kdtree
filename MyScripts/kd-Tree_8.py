#import kdtree
import util
import numpy
import itertools
import collections
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


a=numpy.zeros((10,4))
b=numpy.zeros((10,1))

print a.shape[1]==1
  
fig, ax = plt.subplots() 
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)


#Q = numpy.zeros((numberOfStates,numberOfActions))
Q = numpy.zeros((10,4))
Q[1][0]=0.1
Q[1][1]=0.2
Q[1][2]=0.3
Q[1][3]=0.4

norm = colors.Normalize(Q.min(), Q.max())

x_min = 0
y_min = 0 
x_max = 1
y_max = 1
    
state = 5    
color = [cm.ocean(norm( float(q) )) for q in Q[1]]    

# ---------------------------------------------------------
x_top = (x_min + x_max)/2.0
y_top = (y_min + y_max)/2.0

triangle_coord = [[x_min, y_min, y_min, y_max],[x_min, y_max, x_max, y_max],[x_max, y_max, x_max, y_min, y_max],[x_max, y_min, x_min, y_min]] 
         
assert len(triangle_coord)==len(color), "only four actions/colors supported"
for col, coord in zip(color, triangle_coord):
    print col
    verts = [
        (coord[0], coord[1]), # left, bottom
        (coord[2], coord[3]), # left, top
        (x_top, y_top), # triangle peak, center of square 
        (coord[0], coord[1]), # ignored
        ]
      
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]
      
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=col, lw=2)
    print patch
    ax.add_patch(patch)
    #----------------------------------------------------------

plt.show()





