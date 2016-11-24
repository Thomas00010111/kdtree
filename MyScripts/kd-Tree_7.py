import kdtree
import util
import numpy
import itertools
import collections
import time

nodes = []
points = numpy.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
#util.splitN(points, 0, 0, 16, nodes)
util.splitN(points, 0, 0, 4, nodes)

#print "nodes:", nodes  
print "Number of node: ", len(nodes)
tree = kdtree.create(nodes)

#util.activate(tree, 16)

kdtree.visualize(tree)

print "done"
