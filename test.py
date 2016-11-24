#!/usr/bin/env python

from __future__ import absolute_import

import sys
import random
import logging
import unittest
import doctest
import collections
from itertools import islice
import util
import numpy

try:
    import coverage
    coverage.erase()
    coverage.start()

except ImportError:
    coverage = None

# import after starting coverage, to ensure that import-time code is covered
import kdtree

class RemoveTest(unittest.TestCase):


    def test_remove_duplicates(self):
        """ creates a tree with only duplicate points, and removes them all """

        points = [(1,1)] * 100
        tree = kdtree.create(points)
        self.assertTrue(tree.is_valid())

        random.shuffle(points)
        while points:
            point1 = points.pop(0)

            tree = tree.remove(point1)

            # Check if the Tree is valid after the removal
            self.assertTrue(tree.is_valid())

            # Check if the removal reduced the number of nodes by 1 (not more, not less)
            remaining_points = len(points)
            nodes_in_tree = len(list(tree.inorder()))
            self.assertEqual(nodes_in_tree, remaining_points)


    def test_remove(self, num=100):
        """ Tests random removal from a tree, multiple times """

        for i in range(num):
            self.do_random_remove()


    def do_random_remove(self):
        """ Creates a random tree, removes all points in random order """

        points = list(set(islice(random_points(), 0, 20)))
        tree =  kdtree.create(points)
        self.assertTrue(tree.is_valid())

        random.shuffle(points)
        while points:
            point1 = points.pop(0)

            tree = tree.remove(point1)

            # Check if the Tree is valid after the removal
            self.assertTrue(tree.is_valid())

            # Check if the point1 has actually been removed
            self.assertTrue(point1 not in [n.data for n in tree.inorder()])

            # Check if the removal reduced the number of nodes by 1 (not more, not less)
            remaining_points = len(points)
            nodes_in_tree = len(list(tree.inorder()))
            self.assertEqual(nodes_in_tree, remaining_points)

    def test_remove_empty_tree(self):
        tree = kdtree.create(dimensions=2)
        tree.remove( (1, 2) )
        self.assertFalse(bool(tree))


class AddTest(unittest.TestCase):

    def test_add(self, num=10):
        """ Tests random additions to a tree, multiple times """

        for i in range(num):
            self.do_random_add()


    def do_random_add(self, num_points=100):

        points = list(set(islice(random_points(), 0, num_points)))
        tree = kdtree.create(dimensions=len(points[0]))
        for n, point1 in enumerate(points, 1):

            tree.add(point1)

            self.assertTrue(tree.is_valid())

            self.assertTrue(point1 in [node.data for node in tree.inorder()])

            nodes_in_tree = len(list(tree.inorder()))
            self.assertEqual(nodes_in_tree, n)


class InvalidTreeTests(unittest.TestCase):


    def test_invalid_child(self):
        """ Children on wrong subtree invalidate Tree """
        child = kdtree.KDNode( (3, 2) )
        child.axis = 2
        tree = kdtree.create([(2, 3)])
        tree.left=child
        self.assertFalse(tree.is_valid())

        tree = kdtree.create([(4, 1)])
        tree.right=child
        self.assertFalse(tree.is_valid())


    def test_different_dimensions(self):
        """ Can't create Tree for Points of different dimensions """
        points = [ (1, 2), (2, 3, 4) ]
        self.assertRaises(ValueError, kdtree.create, points)


class TreeTraversals(unittest.TestCase):

    def test_same_length(self):
        tree = random_tree()

        inorder_len = len(list(tree.inorder()))
        preorder_len = len(list(tree.preorder()))
        postorder_len = len(list(tree.postorder()))

        self.assertEqual(inorder_len, preorder_len)
        self.assertEqual(preorder_len, postorder_len)
        
    def test_level_node_unbalanced(self):
        tree2=kdtree.create([ (1, 2), (2, 3) ])
        node=tree2.search_nn((1.1,2.1))
        node[0].add((1,1))
        self.assertEqual(tree2.level(tree2), 0)
        self.assertEqual(node[0].level(tree2), 1)
        self.assertEqual(node[0].left.level(tree2), 2)



class BalanceTests(unittest.TestCase):


    def test_rebalance(self):

        tree = random_tree(1)
        while tree.is_balanced:
            tree.add(random_point())

        tree = tree.rebalance()
        self.assertTrue(tree.is_balanced)



class NearestNeighbor(unittest.TestCase):

    def test_search_knn(self):
        points = [(50, 20), (51, 19), (1, 80)]
        tree = kdtree.create(points)
        point1 = (48, 18)

        all_dist = []
        for p in tree.inorder():
            dist = p.dist(point1)
            all_dist.append([p, dist])

        all_dist = sorted(all_dist, key = lambda n:n[1])

        result = tree.search_knn(point1, 1)
        self.assertEqual(result[0][1], all_dist[0][1])

        result = tree.search_knn(point1, 2)
        self.assertEqual(result[0][1], all_dist[0][1])
        self.assertEqual(result[1][1], all_dist[1][1])

        result = tree.search_knn(point1, 3)
        self.assertEqual(result[0][1], all_dist[0][1])
        self.assertEqual(result[1][1], all_dist[1][1])
        self.assertEqual(result[2][1], all_dist[2][1])

    def test_search_nn(self, nodes=100):
        points = list(islice(random_points(), 0, nodes))
        tree = kdtree.create(points)
        point1 = random_point()

        nn, dist = tree.search_nn(point1)
        best, best_dist = self.find_best(tree, point1)
        self.assertEqual(best_dist, dist, msg=', '.join(repr(p) for p in points) + ' / ' + repr(point1))


    def test_search_nn2(self):
        points = [(1,2,3),(5,1,2),(9,3,4),(3,9,1),(4,8,3),(9,1,1),(5,0,0),
                  (1,1,1),(7,2,2),(5,9,1),(1,1,9),(9,8,7),(2,3,4),(4,5,4.01)]
        tree = kdtree.create(points)
        point1 = (2,5,6)

        nn, dist = tree.search_nn(point1)
        best, best_dist = self.find_best(tree, point1)
        self.assertEqual(best_dist, dist)


    def test_search_nn3(self):
        points = [(0, 25, 73), (1, 91, 85), (1, 47, 12), (2, 90, 20),
      (2, 66, 79), (2, 46, 27), (4, 48, 99), (5, 73, 64), (7, 42, 70),
      (7, 34, 60), (8, 86, 80), (10, 27, 14), (15, 64, 39), (17, 74, 24),
      (18, 58, 12), (18, 58, 5), (19, 14, 2), (20, 88, 11), (20, 28, 58),
      (20, 79, 48), (21, 32, 8), (21, 46, 41), (22, 6, 4), (22, 42, 68),
      (22, 62, 42), (24, 70, 96), (27, 77, 57), (27, 47, 39), (28, 61, 19),
      (30, 28, 22), (34, 13, 85), (34, 39, 96), (34, 90, 32), (39, 7, 45),
      (40, 61, 53), (40, 69, 50), (41, 45, 16), (41, 15, 44), (42, 40, 19),
      (45, 6, 68), (46, 79, 91), (47, 91, 86), (47, 50, 24), (48, 57, 64),
      (49, 21, 72), (49, 87, 21), (49, 41, 62), (54, 94, 32), (56, 14, 54),
      (56, 93, 2), (58, 34, 44), (58, 27, 42), (59, 62, 80), (60, 69, 69),
      (61, 67, 35), (62, 31, 50), (63, 9, 93), (63, 46, 95), (64, 31, 2),
      (64, 2, 36), (65, 23, 96), (66, 94, 69), (67, 98, 10), (67, 40, 88),
      (68, 4, 15), (68, 1, 6), (68, 88, 72), (70, 24, 53), (70, 31, 87),
      (71, 95, 26), (74, 80, 34), (75, 59, 99), (75, 15, 25), (76, 90, 99),
      (77, 75, 19), (77, 68, 26), (80, 19, 98), (82, 90, 50), (82, 87, 37),
      (84, 88, 59), (85, 76, 61), (85, 89, 20), (85, 64, 64), (86, 55, 92),
      (86, 15, 69), (87, 48, 46), (87, 67, 47), (89, 81, 65), (89, 87, 39),
      (89, 87, 3), (91, 65, 87), (94, 37, 74), (94, 20, 92), (95, 95, 49),
      (96, 15, 80), (96, 27, 39), (97, 87, 32), (97, 43, 7), (98, 78, 10),
      (99, 64, 55)]

        tree = kdtree.create(points)
        point1 = (66, 54, 29)

        nn, dist = tree.search_nn(point1)
        best, best_dist = self.find_best(tree, point1)
        self.assertEqual(best_dist, dist)



    def find_best(self, tree, point1):
        best = None
        best_dist = None
        for p in tree.inorder():
            dist = p.dist(point1)
            if best is None or dist < best_dist:
                best = p
                best_dist = dist
        return best, best_dist

    def test_search_nn_dist(self):
        """ tests search_nn_dist() according to bug #8 """

        points = [(x,y) for x in range(10) for y in range(10)]
        tree = kdtree.create(points)
        nn = tree.search_nn_dist((5,5), 2.5)

        self.assertEquals(len(nn), 4)
        self.assertTrue( kdtree.KDNode(data=(6,6)) in nn)
        self.assertTrue( (5,5) in nn)
        self.assertTrue( (5,6) in nn)
        self.assertTrue( (6,5) in nn)


    def test_search_nn_dist_random(self):

        for n in range(50):
            tree = random_tree()
            point1 = random_point()
            points = tree.inorder()

            points = sorted(points, key=lambda p: p.dist(point1))

            for p in points:
                dist = p.dist(point1)
                nn = tree.search_nn_dist(point1, dist)

                for pn in points:
                    if pn in nn:
                        self.assertTrue(pn.dist(point1) < dist, '%s in %s but %s < %s' % (pn, nn, pn.dist(point1), dist))
                    else:
                        self.assertTrue(pn.dist(point1) >= dist, '%s not in %s but %s >= %s' % (pn, nn, pn.dist(point1), dist))


class PointTypeTests(unittest.TestCase):
    """ test using different types as points """

    def test_point_types(self):
        emptyTree = kdtree.create(dimensions=3)
        point1 = (2, 3, 4)
        point2 = [4, 5, 6]
        Point = collections.namedtuple('Point', 'x y z')
        point3 = Point(5, 3, 2)
        tree = kdtree.create([point1, point2, point3])
        res, dist = tree.search_nn( (1, 2, 3) )

        self.assertEqual(res, kdtree.KDNode( (2, 3, 4) ))


class PayloadTests(unittest.TestCase):
    """ test tree.add() with payload """

    def test_payload(self, nodes=100, dimensions=3):
        points = list(islice(random_points(dimensions=dimensions), 0, nodes))
        tree = kdtree.create(dimensions=dimensions)

        for i, p in enumerate(points):
            tree.add(p).payload = i

        for i, p in enumerate(points):
            self.assertEqual(i, tree.search_nn(p)[0].payload)
            
class SplitNodeTest(unittest.TestCase):
#     def test_tree_size(self):
#         listSplitPoints = []
#         points = numpy.array([[0.0, 0.0],[0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
#         util.splitN(points, 0,0,5, listSplitPoints)
#         tree2dN = kdtree.createNewTree(listSplitPoints)
#         kdtree.visualize(tree2dN)
#         
#         print "len: ", len(kdtree.level_order(tree2dN))
        
        
    def test_splitNode(self):
        ''' find the best matching node and split it, then find the best matching node again. 
            Check if point lies in new generated node'''
        print "---------- test splitNode --------"
        #create tree with 2 levels
        listSplitPoints = []
        points = numpy.array([[0.0, 0.0],[0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0,0,5, listSplitPoints)
        tree2dN = kdtree.createNewTree(listSplitPoints)
        util.activate(tree2dN, 2)
        
        #points
        point1 = [0.9,0.1]
        point2 = [0.1,0.9]
        
        kdtree.visualize(tree2dN)

        # split
        print "found: ", tree2dN.get_path_to_best_matching_node(point1)[-1] 
        tree2dN.get_path_to_best_matching_node(point1)[-1].activate_subnodes()
        kdtree.visualize(tree2dN)
        tree2dN.get_path_to_best_matching_node(point1)[-1].activate_subnodes()
        kdtree.visualize(tree2dN)
        print "data: ",  tree2dN.get_path_to_best_matching_node(point1)[-1].data
        self.assertEqual( tree2dN.get_path_to_best_matching_node(point1)[-1].data, [0.875, 0.125], "wrong node")
        
        tree2dN.get_path_to_best_matching_node(point2)[-1].activate_subnodes()
        tree2dN.get_path_to_best_matching_node(point2)[-1].activate_subnodes()
        self.assertEqual( tree2dN.get_path_to_best_matching_node(point2)[-1].data, [0.125, 0.875], "wrong node")
        del tree2dN
        
    def test_getNode(self):
        print "---------- test getNode --------"
        listSplitPoints = []
        points = numpy.array([[0.0, 0.0],[0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0,0,6, listSplitPoints)
        tree = kdtree.createNewTree(listSplitPoints)
        util.activate(tree, 6)
        kdtree.visualize(tree)
        nodeLabel = 117
        node = kdtree.getNode(tree, nodeLabel)
        self.assertEqual( node.label, nodeLabel, "returned wrong node")
        del tree


class TreeCreationTest(unittest.TestCase):
    def test_numberOfNodes(self):
        highestlevel = 4
        numberOfStates= 2**(highestlevel+2)-1
        no1dN = []
        points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0, 0, highestlevel, no1dN)
        tree = kdtree.createNewTree(no1dN)
        self.assertEqual(tree.getHighestNodeId, numberOfStates, "created and expected number of states does not match")
        
        
    def test_numberOfActiveStates(self):
        """only temporary, active property will disapear in future"""
        highestlevel = 4
        numberOfStates= 2**(highestlevel+2)-1
        no1dN = []
        points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0, 0, highestlevel, no1dN)
        tree = kdtree.createNewTree(no1dN)

        util.activate(tree, highestlevel+1)
         
        activeNodes = len([n for n in kdtree.level_order(tree) if n.active])
        print "activeNodes: ", activeNodes, "       numberOfStates: ", numberOfStates
        self.assertEqual(activeNodes, numberOfStates, "not the correct number of nodes active")
        
        
        
        
#     def test_numberOfActiveNodes(self):
#         highestlevel = 4
#         numberOfStates= 2**(highestlevel+2) -1
#         no1dN = []
#         points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
#         util.splitN(points, 0, 0, highestlevel, no1dN)
#         tree = kdtree.createNewTree(no1dN)
#         self.assertEqual(tree.getHighestNodeId, numberOfStates, "created and expected number of states does not match")
#         util.activate(tree, highestlevel+1)
#         
#         kdtree.visualize(tree)

    
    def test_createAndLableTree(self):
        ''' Create new tree with new ids starting from 0'''
        print "---------- test createAndLabelTree 1--------"
        no1dN = []
        points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0, 0, 6, no1dN)
        tree1 = kdtree.createNewTree(no1dN)
        
        label=0
        for n in kdtree.level_order(tree1):
            self.assertIsNotNone(kdtree.getNode(tree1, label), "1: node with label: "+ str(label) + " not found in tree")
            label+=1
        kdtree.visualize(tree1)   
            
        print "---------- test createAndLabelTree 2--------"
        no2dN = []
        points = numpy.array([[0.0, 0.01], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0, 0, 6, no2dN)
        tree2 = kdtree.createNewTree(no2dN)
        kdtree.visualize(tree2)  
        
        label=0
        for n in kdtree.level_order(tree2):
            self.assertIsNotNone(kdtree.getNode(tree2, label), "2: node with label: "+ str(label) + " not found in tree")
            label+=1
  
        self.assertNotEqual(tree1, tree2, "trees have to be different")
        
class DisplayTreeTest(unittest.TestCase):
    """ This test only works when the active=False"""
    def test_showQ(self):
        import matplotlib.pyplot as plt
        import time
          
        print "---------- DisplayTreeTest ----------"
        #plt.figure(self.fig_values.number)
        maxLevel=2
        no1dN = []
        points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0, 0, maxLevel, no1dN)
        tree = kdtree.createNewTree(no1dN)
          
        numberOfStates= tree.getHighestNodeId
        numberOfActions = 4
        
        kdtree.visualize(tree)
        Q = numpy.ones((100,numberOfActions))
        
        n = tree.get_path_to_best_matching_node([0.75, 0.75])[-1]
        print n.label
        n.split2([0.85, 0.75], axis=0, sel_axis = (lambda axis: axis))
        kdtree.visualize(tree)  
        # only leaves are shown!
        # States are positioned like in the tree i.e. xy axes splits in tree represent xy position in coord system
        # 0, 0 is bottom left, x increases to the right 
        # action 0 is to the left
        # Q[State][action]
        Q[3][0] = 0 # bottom left, action left
#         Q[5][1] = 0.1 # above Q[2] (in y direction), right
#         Q[58][2] = 0.1 #right top corner, down
#         Q[4][0] = 0.5
        kdtree.plotQ2D(tree, min_coord=[0, 0], max_coord=[1, 1],Values = Q, plt=plt, plot="Q")
        time.sleep(5)  
          
        

        
        
        
# class DisplayTreeTest(unittest.TestCase):
#     """ This test only works when the active=False"""
#     def test_showQ(self):
#         import matplotlib.pyplot as plt
#         import time
#          
#         print "---------- DisplayTreeTest ----------"
#         #plt.figure(self.fig_values.number)
#         plt.title("Values")
#         maxLevel=4
#         no1dN = []
#         points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
#         util.splitN(points, 0, 0, maxLevel, no1dN)
#         tree = kdtree.createNewTree(no1dN)
#          
#         numberOfStates= tree.getHighestNodeId
#         numberOfActions = 4
#          
#         util.activate(tree, maxLevel)
#         kdtree.visualize(tree)
#         Q = numpy.ones((numberOfStates,numberOfActions))
#          
#         # only leaves are shown!
#         # States are positioned like in the tree i.e. xy axes splits in tree represent xy position in coord system
#         # 0, 0 is bottom left, x increases to the right 
#         # action 0 is to the left
#         # Q[State][action]
#         Q[2][0] = 0 # bottom left, action left
#         Q[5][1] = 0.1 # above Q[2] (in y direction), right
#         Q[58][2] = 0.1 #right top corner, down
# #         Q[4][0] = 0.5
#         kdtree.plotQ2D(tree, min_coord=[0, 0], max_coord=[1, 1],Values = Q, plt=plt, plot="Q")
#         time.sleep(5)  
    
    
#     def test_showQ(self):
#         import matplotlib.pyplot as plt
#         import time
#          
#         print "---------- DisplayTreeTest ----------"
#         #plt.figure(self.fig_values.number)
#         plt.title("Values")
#         levels=4
#         numberOfStates= 2**(levels+1)
#         numberOfActions = 4
#         no1dN = []
#         points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
#         util.splitN(points, 0, 0, levels, no1dN)
#         tree = kdtree.createNewTree(no1dN)
#         kdtree.visualize(tree)
#         util.activate(tree, levels)
#         Q = numpy.ones((numberOfStates,numberOfActions))
#          
#         # only leaves are shown!
#         # States are positioned like in the tree i.e. xy axes splits in tree represent xy position in coord system
#         # 0, 0 is bottom left, x increases to the right 
#         # action 0 is to the left
#         # Q[State][action]
#         Q[0][0] = 0 # bottom left, action left
#         Q[1][1] = 0.1 # above Q[0] (in y direction), right
#         Q[26][2] = 0.1 #right top corner, down
# #         Q[3][0] = 0.3
# #         Q[4][0] = 0.5
#         kdtree.plotQ2D(tree, min_coord=[0, 0], max_coord=[1, 1],Values = Q, plt=plt, plot="Q")
#         time.sleep(2)      
        

def random_tree(nodes=20, dimensions=3, minval=0, maxval=100):
    points = list(islice(random_points(), 0, nodes))
    tree = kdtree.create(points)
    return tree

def random_point(dimensions=3, minval=0, maxval=100):
    return tuple(random.randint(minval, maxval) for _ in range(dimensions))

def random_points(dimensions=3, minval=0, maxval=100):
    while True:
        yield random_point(dimensions, minval, maxval)


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    suite.addTest(doctest.DocTestSuite(kdtree))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    if coverage is not None:
        coverage.stop()
        coverage.report([kdtree])
        coverage.erase()
