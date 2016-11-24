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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# 
# try:
#     import coverage
#     coverage.erase()
#     coverage.start()
# 
# except ImportError:
#     coverage = None

# import after starting coverage, to ensure that import-time code is covered
import kdtree


class CreateUnconstraintTree(unittest.TestCase):

    def test_createTreeRoot(self):
        '''
        Created tree should have a root and two empty nodes with a label'''
        sel_axis = (lambda axis: axis)
        tree = kdtree.createNewTree([[0.5, 0.5, 0.5]],axis = 1, sel_axis= sel_axis)
#        kdtree.visualize(tree)
        self.assertTrue(tree.label == 2, "left label is not 2")
        self.assertTrue(tree.axis == 1, "left label is not 2")

        self.assertTrue(tree.left is not None, "left node of root is missing")
        self.assertTrue(tree.left.label == 0, "left label is not 0")
        
        self.assertTrue(tree.right is not None, "right node of root is missing")
        self.assertTrue(tree.right.label == 1, "right label is not 1")

    def test_createTree(self):
        '''
        Created tree and split left and right node'''
        sel_axis = (lambda axis: axis)
        tree = kdtree.createNewTree([[0.5, 0.5, 0.5]],axis = 1, sel_axis= sel_axis)
        
        #add right node
        point_right = [0.4, 0.5, 0.5]
        tree.split2(point_right, axis = 2)
        self.assertTrue(tree.right.data == point_right, "right node data not set")
        self.assertTrue(tree.right.axis == 2, "right node axis not set")
        self.assertTrue(tree.right.label == 1, "right node label wrong")
        self.assertTrue(tree.right.right is not None, "right node chikd missing")
        self.assertTrue(tree.right.left is not None, "right node chikd missing")
        
        #add left node
        point_left = [0.4, 0.4, 0.5]
        tree.split2(point_left, axis = 2)
        self.assertTrue(tree.left.data == point_left, "left node data not set")
        self.assertTrue(tree.left.axis == 2, "left node axis not set")
        self.assertTrue(tree.left.label == 0, "left node label wrong")
        self.assertTrue(tree.left.left is not None, "left node chikd missing")
        self.assertTrue(tree.left.left is not None, "left node chikd missing")

        
  
    def test_add_empty_nodes_with_label_when_splitting(self):
        """
        When a node is split along a certain axis, then this split should be active immediately.
        Create tree, split right and left node, try to find matching node
        """
        print "----- test_add_empty_nodes_with_label_when_splitting -----"
        sel_axis = (lambda axis: axis)
        
        #create tree, first node splits in x direction
        tree = kdtree.createNewTree([[0.5, 0.5]],axis = 0, sel_axis= sel_axis)
        kdtree.visualize(tree)
        
        point_left = [0.4, 0.5]
        tree.split2(point_left, axis = 0)
        kdtree.visualize(tree)
         
        point1 = [0.3, 0.5]
        found_node = tree.get_path_to_leaf(point1)[-1]
        correct_node1 = 3
        self.assertEqual(found_node.label, correct_node1, "Not correct node found")
        
        point_right = [0.6, 0.5]
        tree.split2(point_right, axis = 1)
        kdtree.visualize(tree)
        
        point2 = [0.6, 0.7]
        found_node = tree.get_path_to_leaf(point2)[-1]
        correct_node2 = 6
        self.assertEqual(found_node.label, correct_node2, "Not correct node found")
                
        print "----- end: test_add_empty_nodes_with_label_when_splitting -----"
        
       
    
    def test_compare_old_to_new_method_to_create_trees(self):
        """ 
        tree created with old method should be equal to tree created with new method
        """
        nodes = util.generate_sequence_of_points(2, 2)
        tree1 = kdtree.createNewTree(nodes)
        kdtree.visualize(tree1)
        
        sel_axis = (lambda axis: axis)
        tree2 = kdtree.createNewTree([[0.5, 0.5]],axis = 0, sel_axis= sel_axis)
        tree2.split2([0.25, 0.5], axis = 1)
        tree2.split2([0.75, 0.5], axis = 1)
        
        #left
        tree2.split2([0.25, 0.25], axis = 0, sel_axis = sel_axis)
        tree2.split2([0.25, 0.75], axis = 0, sel_axis = sel_axis)
         
        #right
        tree2.split2([0.75, 0.25], axis = 0, sel_axis = sel_axis)
        tree2.split2([0.75, 0.75], axis = 0, sel_axis = sel_axis)
        
        kdtree.visualize(tree2)
        
        for n in zip(kdtree.level_order(tree1), kdtree.level_order(tree2)):
            self.assertEqual(n[0].data, n[1].data, "elements not equal")
            
            if n[0].data is not None and n[1].data is not None:
                self.assertEqual(n[0].axis, n[1].axis, "elements not equal")
        
        
        
    def test_plotTree(self):
        # function to chose next spillting axis
        sel_axis = (lambda axis: axis)
        
        #create tree, first node splits in x direction
        tree = kdtree.createNewTree([[0.5, 0.5]],axis = 0, sel_axis= sel_axis)
        tree.split2([0.4, 0.5], axis = 0, sel_axis = sel_axis)
        
        #add right node root node and left node to new node
        tree.split2([0.6, 0.5], axis = 1, sel_axis = sel_axis)
        tree.split2([0.7, 0.4], axis = 0, sel_axis = sel_axis)
        
        print "node before: ", tree.get_path_to_best_matching_node([0.3, 0.5])[-1].label
        print "node before: ", tree.get_path_to_best_matching_node([0.3, 0.5])[-1].label
        #add a node
        tree.split2([0.3, 0.6], axis = 1, sel_axis = sel_axis)

        print "node after: ", tree.get_path_to_best_matching_node([0.3, 0.5])[-1].label
        print "node after: ", tree.get_path_to_best_matching_node([0.3, 0.5])[-1].label
        
        kdtree.visualize(tree)

#        img=mpimg.imread("test_unconstraint_tree.png")  
#        plt.imshow(img)
 
        # Compare to image test_unconstraint_tree.png
        kdtree.plot2D(tree, plt=plt)
        
        

        
       
#        print "---- end: test plot ----"
        

#         tree.add([0.4, 0.5, 0.25])
#         
#         #tree.add([0.5, 0.5, 0.15])
#         #tree.add([0.5, 0.4, 0.25])
#         kdtree.visualize(tree)
#         
#         point1 = [0.5, 0.5, 0.25]
#         n = tree.get_path_to_best_matching_node(point1)[-1]
#         print n.label
#         n.split2(2, point1)
#         kdtree.visualize(tree)
#         
#         point1 = [0.5, 0.6, 0.1]
#         n = tree.get_path_to_best_matching_node(point1)[-1]
#         print n.label
#         n.split2(2, point1)
#         kdtree.visualize(tree)
        
       