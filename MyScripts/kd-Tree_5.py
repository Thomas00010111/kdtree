import kdtree
import util
import numpy
import itertools
import collections
import networkx as nx
import matplotlib.pyplot as plt

'''Split leave nodes in tree'''


no2dN = []
#points = numpy.array([[0.0, 0.0,  0.0],[0.0 ,  0.0, 1.0], [ 0.0, 1.0, 0.0], [ 0.0, 1.0, 1.0], [1.0, 0.0,  0.0], [1.0, 0.0,  1.0], [1.0, 1.0,  0.0], [1.0, 1.0,  1.0]])
points = numpy.array([[0.0, 0.0],[0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
util.splitN(points, 0,0,5, no2dN)

#print "no3dN:", no3dN  
print "Number of nodes3dN: ", len(no2dN)
point1 = [0.9,0.9]
point2 = [0.9,0.1]
tree2dN = kdtree.create(no2dN)
util.activate(tree2dN, 2)
tree2dN.get_path_to_best_matching_node(point1)[-1].split()
tree2dN.get_path_to_best_matching_node(point1)[-1].split()
print "data:", tree2dN.get_path_to_best_matching_node(point1)[-1].data
print "id:", tree2dN.get_path_to_best_matching_node(point1)[-1].label

tree2dN.get_path_to_best_matching_node(point2)[-1].split()
tree2dN.get_path_to_best_matching_node(point2)[-1].split()
print "data:", tree2dN.get_path_to_best_matching_node(point2)[-1].data
print "id:", tree2dN.get_path_to_best_matching_node(point2)[-1].label

kdtree.visualize(tree2dN)
kdtree.plot2D(tree2dN)

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


