import kdtree
import util
import numpy
import itertools
import collections
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# plt.ioff()
# verts = [
#     (0., 0.), # left, bottom
#     (0., 1.), # left, top
#     (1., 1.), # right, top
#     (1., 0.), # right, bottom
#     (0., 0.), # ignored
#     ]
#  
# codes = [Path.MOVETO,
#          Path.LINETO,
#          Path.LINETO,
#          Path.LINETO,
#          Path.CLOSEPOLY,
#          ]
#  
# path = Path(verts, codes)
#  
# fig = plt.figure()
# ax = fig.add_subplot(111)
# patch = patches.PathPatch(path, facecolor='orange', lw=2)
# ax.add_patch(patch)
# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# plt.show()



no2dN = []
points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
util.splitN(points, 0, 0, 6, no2dN)
tree2dN = kdtree.create(no2dN)
util.activate(tree2dN, 4)
  
kdtree.visualize(tree2dN)
 
V = numpy.random.rand(len(no2dN),1)
kdtree.plotQ2D(tree2dN, Values=V)
#kdtree.plot2D(tree2dN)



#kdtree.plot2DUpdate(tree2dN)
# 
# 
# no3dN_test = []
# points = numpy.array([[0.0, 0.0,  0.0],[0.0 ,  0.0, 1.0], [ 0.0, 1.0, 0.0], [ 0.0, 1.0, 1.0], [1.0, 0.0,  0.0], [1.0, 0.0,  1.0], [1.0, 1.0,  0.0], [1.0, 1.0,  1.0]])
# splitN_test(points, 0,0,7, no3dN_test)
# 
# assert all(x in no3dN_test for x in no3dN), "NOT EQUAL !!!"

#kdtree.plot2D(tree)

# print [ n.label for n in tree.get_path_to_best_matching_node((0.1,0.1))]
# tree.get_path_to_best_matching_node((0.1,0.1))[-1].split()
#  
# print [ n.label for n in tree.get_path_to_best_matching_node((0.1,0.1))]
# tree.get_path_to_best_matching_node((0.1,0.1))[-1].split()
#  
# print [ n.label for n in tree.get_path_to_best_matching_node((0.1,0.1))]
# tree.get_path_to_best_matching_node((0.1,0.1))[-1].split()


# print [ str(n.label) +"  " + str(n.height()) for n in tree.get_path_to_best_matching_node((0.6, 0.4))]
# 
# print [ n.label for n in tree.get_path_to_best_matching_node((0.6, 0.1))]
# 
# print [ n.label for n in tree.get_path_to_best_matching_node((0.175, 0.6))]


# kdtree.visualize(tree)
# 
# 
# # unbalanced tree
# tree2=kdtree.create([ (1, 2), (2, 3) ])
# node=tree2.search_nn((1.1,2.1))
# print node[0].label
# node[0].add((1,1))
# 
# kdtree.visualize(tree2)
# 
# print "tree2.height(): ",  tree2.height()
# print "node[0].level(tree2): ",  node[0].level(tree2)
# print "tree2.level(tree2): ",  tree2.level(tree2)


