import kdtree
import numpy
import itertools
import collections
import networkx as nx
import matplotlib.pyplot as plt




def split(x0, y0, x1, y1, axis, level, maxlevel, nodes):
    if level < maxlevel:
        newNode = numpy.array([(x0+x1)/2.0, (y0+y1)/2.0])
        nodes.append(newNode)
        
        if axis == 0:
            split(x0, y0, (x0+x1)/2.0, y1, 1, level+1, maxlevel, nodes)
            split((x0+x1)/2.0, y0, x1, y1, 1, level+1, maxlevel, nodes)
        
        if axis == 1:
            split(x0, y0, x1, (y0+y1)/2.0, 0, level+1, maxlevel, nodes)
            split(x0, (y0+y1)/2.0, x1, y1, 0, level+1, maxlevel, nodes)
    else:
        return


dimension = (3,3)

no = []
split(0.0, 0.0, 1.0, 1.0, 0, 0, dimension[0], no)

print "no:", no  
print "Number of nodes: ", len(no)

    
tree = kdtree.create(no)

#kdtree.visualize(tree)

#print [ n.label for n in tree.get_path_to_best_matching_node((0.6,0.4))]
print [ str(n.label) +"  " + str(n.height()) for n in tree.get_path_to_best_matching_node((0.6, 0.4))]
print [ n.label for n in tree.get_path_to_best_matching_node((0.6, 0.1))]
print [ n.label for n in tree.get_path_to_best_matching_node((0.175, 0.6))]


kdtree.visualize(tree)


# unbalanced tree
tree2=kdtree.create([ (1, 2), (2, 3) ])
node=tree2.search_nn((1.1,2.1))
print node[0].label
node[0].add((1,1))

kdtree.visualize(tree2)

print "tree2.height(): ",  tree2.height()
print "node[0].level(tree2): ",  node[0].level(tree2)
print "tree2.level(tree2): ",  tree2.level(tree2)


#asumtion no
kdtree.plot2D(tree)
