import kdtree
import util
import numpy
import itertools
import collections
import networkx as nx
import matplotlib.pyplot as plt

'''Split leave nodes in tree'''

MAX_AXIS = 2

def split(x0, y0, x1, y1, axis, level, maxlevel, nodes):
    #max_coord=[[0.0, 1.0], [0.0, 1.0]]
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


def splitN_test(points,level, axis, maxlevel, nodes):
    #max_coord=[[0.0, 1.0], [0.0, 1.0]]
    if level < maxlevel:
        numberOfPoints = points.shape[0];
        newNode = numpy.array(sum(points)/numberOfPoints).tolist()
        nodes.append(newNode)
        
        #sort put larger points in one array
        bigger=numpy.empty(points.shape)
        smaller=numpy.empty(points.shape)
        avg_axis=sum(points[:,axis])/numberOfPoints
       
        index_s=0
        index_b=0
        for p in points:
            if p[axis] <=  avg_axis:
                smaller[index_s] =p
                index_s+=1
            else:
                bigger[index_b] =p
                index_b+=1
        
        assert len(smaller) == len(bigger), "Number of points has to be equal in both lists"        
        #generate new border
        newPoints=[]
        for p in range(0, index_b):
            temp = bigger[p].copy()
            #temp[axis] = bigger[p][axis]/2.0
            temp[axis] = avg_axis
            newPoints.append(temp) 
        
        for n in newPoints:      
            smaller[index_s] =n
            index_s+=1
            bigger[index_b] =n
            index_b+=1
        
        axis=(axis+1)%points.shape[1]
        #print "smaller: ", smaller
        #print "bigger: ", bigger
        splitN_test(smaller, level+1, axis, maxlevel, nodes )
        splitN_test(bigger, level+1, axis, maxlevel, nodes )
#         if axis == 0:
#             split(x0, y0, (x0+x1)/2.0, y1, 1, level+1, maxlevel, nodes)
#             split((x0+x1)/2.0, y0, x1, y1, 1, level+1, maxlevel, nodes)
#         
#         if axis == 1:
#             split(x0, y0, x1, (y0+y1)/2.0, 0, level+1, maxlevel, nodes)
#             split(x0, (y0+y1)/2.0, x1, y1, 0, level+1, maxlevel, nodes)
    else:
        return



dimension = (4,4)
# noN = []
# points = numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
# splitN(points,0, 2,noN )
# print "noN: ", noN
# tree = kdtree.create(noN)
# kdtree.visualize(tree)
# 
# no = []
# 
# 
# no=[[0.5, 0.5, 0.5] ,[0.25, 0.5, 0.5],[0.25, 0.25, 0.5], [0.25, 0.75, 0.5],  [0.75, 0.5, 0.5],[0.75, 0.25, 0.5],[0.75, 0.75, 0.5] ]
# 
# 
# print "no:", no  
# print "Number of nodes: ", len(no)
#     
# tree = kdtree.create(no)
# kdtree.visualize(tree)


# no2dN = []
# points = numpy.array([[0.0, 0.0],[0.0 , 1.0], [1.0, 0.0], [1.0, 1.0]])
# util.splitN(points, 0,0,2, no2dN)
# print "no2dN:", no2dN  
# print "Number of nodes2dN: ", len(no2dN)
# tree2dN = kdtree.create(no2dN)
# kdtree.visualize(tree2dN)
# kdtree.plot2D(tree2dN)

no3dN = []
#points = numpy.array([[0.0, 0.0,  0.0],[0.0 ,  0.0, 1.0], [ 0.0, 1.0, 0.0], [ 0.0, 1.0, 1.0], [1.0, 0.0,  0.0], [1.0, 0.0,  1.0], [1.0, 1.0,  0.0], [1.0, 1.0,  1.0]])
points = numpy.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
util.splitN(points, 0,0,5, no3dN)

#print "no3dN:", no3dN  
print "Number of nodes3dN: ", len(no3dN)
point = [1,0,0,0]
tree3dN = kdtree.create(no3dN)
util.activate(tree3dN, 2)
print tree3dN.get_path_to_best_matching_node(point)[-1].label
kdtree.visualize(tree3dN)
#kdtree.plot2D(tree3dN)
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


