import kdtree
import pickle
import itertools
import numpy
import util
import unittest
import dill # to pickle lambda functions
import random

class RemoveTest(unittest.TestCase):
    def test_create_tree_create_new_tree_with_data_from_first(self):
        nodes = []
        netDimension = 2
        levels = 3
        sequence = ["".join(seq) for seq in itertools.product("01", repeat=netDimension)]
        points_temp= numpy.array([list(s) for s in sequence])
        points = numpy.array([map(float, f) for f in points_temp])
        
        util.splitN(points, 0, 0, levels, nodes)
        
        #print "nodes:", nodes  
        tree = kdtree.createNewTree(nodes)
#        kdtree.visualize(tree)
         
        points_tree =  [(d.data, d.axis) for d in kdtree.level_order(tree) if d.data is not None]
        points_tree_copy = list(points_tree)
        
        tree1 = kdtree.createNewTree([d[0] for d in points_tree]) # points_tree1 is changed
#        kdtree.visualize(tree1)       
        
        points_tree2 =  [(d.data, d.axis) for d in kdtree.level_order(tree1) if d.data is not None]
        numpy.testing.assert_array_equal(points_tree_copy, points_tree2, "trees not equal?")
        

        
    def test_add_node_and_pickle_tree(self):
        print "-------------- test_pickle_tree ---------------"
        nodes = []
        netDimension = 3
        levels = 3
        sequence = ["".join(seq) for seq in itertools.product("01", repeat=netDimension)]
        points_temp= numpy.array([list(s) for s in sequence])
        points = numpy.array([map(float, f) for f in points_temp])
        
        util.splitN(points, 0, 0, levels, nodes)
        
        tree = kdtree.createNewTree(nodes)
        
        for i in range(10):
            tree.split2([random.random(), random.random(), random.random()], axis=random.randint(0, netDimension-1))
        
        points_tree =  [(d.data, d.axis) for d in kdtree.level_order(tree) if d.data is not None]
        kdtree.visualize(tree)
        
        kdtree.save( tree, "save_tree_test.pkl" )
        tree_loaded = kdtree.load("save_tree_test.pkl")
        
        points_tree_loaded =  [(d.data, d.axis) for d in kdtree.level_order(tree_loaded) if d.data is not None]
        
        numpy.testing.assert_array_equal(points_tree, points_tree_loaded, "trees not equal?") 
        
