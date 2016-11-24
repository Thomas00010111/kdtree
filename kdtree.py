
"""A Python implemntation of a kd-tree

This package provides a simple implementation of a kd-tree in Python.
https://en.wikipedia.org/wiki/K-d_tree
"""

from __future__ import print_function

import operator
import math
from collections import deque
from functools import wraps
from timeit import itertools
import matplotlib.pyplot as plt_kd
import datetime
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy
import dill
import pickle

id_label = 0
#fig_kd, ax1 = plt_kd.subplots() 
        
itertools.count(0)


__author__ = u'Stefan Kögl <stefan@skoegl.net>'
__version__ = '0.12'
__website__ = 'https://github.com/stefankoegl/kdtree'
__license__ = 'ISC license'


# maps child position to its comparison operator
COMPARE_CHILD = {
    0: (operator.le, operator.sub),
    1: (operator.ge, operator.add),
}


class Node(object):
    """ A Node in a kd-tree

    A tree is represented by its root node, and every node represents
    its subtree"""

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


    @property
    def is_leaf(self):
        """ Returns True if a Node has no subnodes

        >>> Node().is_leaf
        True

        >>> Node( 1, left=Node(2) ).is_leaf
        False
        """
        return (not self.data) or \
               (all(not bool(c) for c, p in self.children))


    def preorder(self):
        """ iterator for nodes: root, left, right """

        if not self:
            return

        yield self

        if self.left:
            for x in self.left.preorder():
                yield x

        if self.right:
            for x in self.right.preorder():
                yield x


    def inorder(self):
        """ iterator for nodes: left, root, right """

        if not self:
            return

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inorder():
                yield x


    def postorder(self):
        """ iterator for nodes: left, right, root """

        if not self:
            return

        if self.left:
            for x in self.left.postorder():
                yield x

        if self.right:
            for x in self.right.postorder():
                yield x

        yield self


    @property
    def children(self):
        """
        Returns an iterator for the non-empty children of the Node

        The children are returned as (Node, pos) tuples where pos is 0 for the
        left subnode and 1 for the right.

        >>> len(list(create(dimensions=2).children))
        0

        >>> len(list(create([ (1, 2) ]).children))
        0

        >>> len(list(create([ (2, 2), (2, 1), (2, 3) ]).children))
        2
        """

        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1


    def set_child(self, index, child):
        """ Sets one of the node's children

        index 0 refers to the left, 1 to the right child """

        if index == 0:
            self.left = child
        else:
            self.right = child
            
    def level(self, tree):
        return tree.height() - self.height()


    def height(self):
        """
        Returns height of the (sub)tree, without considering
        empty leaf-nodes

        >>> create(dimensions=2).height()
        0

        >>> create([ (1, 2) ]).height()
        1

        >>> create([ (1, 2), (2, 3) ]).height()
        2
        """

        min_height = int(bool(self))
        return max([min_height] + [c.height()+1 for c, p in self.children])


    def get_child_pos(self, child):
        """ Returns the position if the given child

        If the given node is the left child, 0 is returned. If its the right
        child, 1 is returned. Otherwise None """

        for c, pos in self.children:
            if child == c:
                return pos


    def __repr__(self):
        return '<%(cls)s - %(data)s>' % \
            dict(cls=self.__class__.__name__, data=repr(self.data))


    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


def require_axis(f):
    """ Check if the object of the function has axis and sel_axis members """

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                    'to have an axis and a sel_axis function' %
                    dict(func_name=f.__name__, node=repr(self)))

        return f(self, *args, **kwargs)

    return _wrapper



class KDNode(Node):
    """ A Node that contains kd-tree specific data and methods """
    #newid = itertools.count().next

    def __init__(self, data=None, left=None, right=None, axis=None,
            sel_axis=None, dimensions=None, label=None):
        """ Creates a new node for a kd-tree

        If the node will be used within a tree, the axis and the sel_axis
        function should be supplied.

        sel_axis(axis) is used when creating subnodes of the current node. It
        receives the axis of the parent node and returns the axis of the child
        node. """
        global id_label
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions
        
        if data is not None:
            self.dimensions = len(data)
#        self._active = False
        self._active = True
        
        if label is None:
            self.label = id_label
            id_label+=1
        else:
            self.label = label
#         if not data==None:
#             #self.label = KDNode.newid()
#             self.label = id_label
#             #print (self.label)  
#             id_label+=1
#         else:
#             self.label=None
           
        #print ("KDNode self.label: ", self.label)

    def __eq__(self, other):
        ''' !!!! Added assert to see if this has influence on my program !!!!''' 
#        return self.label==other.label
        assert False,"Changed method see commented out line!! Now test are ok, but does it have influence on my program?"
        return self.data == other
    
    @property
    def getHighestNodeId(self):
        global id_label
        return id_label
    
    @property
    def active(self):
        return self._active

    @active.setter
    def active(self,state):
        self._active = state
        
    def split(self):
        assert False, "Renamed to activate_subnodes() because the method does not split, it assumes existing tree"
    
    def activate_subnodes(self):
        '''split node and return label of two new nodes'''
        print ("splitting: ", self.label)
        self.left.active = True
        self.right.active = True
        return self.left.label, self.right.label 
    
    @require_axis
    def add(self, point1):
        """
        Adds a point1 to the current node or iteratively
        descends to one of its children.

        Users should call add() only to the topmost tree.
        """
        return self.split2(point1, self.axis, self.sel_axis)
    
    
    @require_axis
    def split2(self, point1, axis=None, sel_axis = None):
        global id_label
        current = self
        while True:
            check_dimensionality([point1], dimensions=current.dimensions)

            # Adding has hit an empty leaf-node, add here
            if current.data is None:
                current.axis = axis
#                current.label = id_label
#                id_label += 1
                current.data = point1
                current.left = self.__class__(data=None, axis=None, sel_axis=sel_axis, dimensions=current.dimensions)
                current.right = self.__class__(data=None, axis=None, sel_axis=sel_axis, dimensions=current.dimensions)
                return current

            # split on self.axis, recurse either left or right
            if point1[current.axis] < current.data[current.axis]:
                if current.left is None:    #is None checks reference not __nonzero__
                    current.left = current.create_subnode(point1, axis, sel_axis)
                    return current.left
                else:
                    current = current.left
                    #self.left.add(point1)
            else:
                if current.right is None:
                    current.right = current.create_subnode(point1, axis, sel_axis)
                    return current.right
                else:
                    current = current.right


    @require_axis
    def create_subnode(self, data, axis=None, sel_axis = None):
        """ Creates a subnode for the current node """
        left = self.__class__(data=None,
                axis=None,
                sel_axis=sel_axis,
                dimensions=self.dimensions)
        right = self.__class__(data=None,
                axis=None,
                sel_axis=sel_axis,
                dimensions=self.dimensions)
        
        return self.__class__(data,left=left, right=right,
                axis=self.sel_axis(axis),
                sel_axis=sel_axis,
                dimensions=self.dimensions)


    @require_axis
    def find_replacement(self):
        """ Finds a replacement for the current node

        The replacement is returned as a
        (replacement-node, replacements-parent-node) tuple """

        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)

        return (child, parent if parent is not None else self)


    def should_remove(self, point1, node):
        """ checks if self's point1 (and maybe identity) matches """
        if not self.data == point1:
            return False

        return (node is None) or (node is self)


    @require_axis
    def remove(self, point1, node=None):
        """ Removes the node with the given point1 from the tree

        Returns the new root node of the (sub)tree.

        If there are multiple points matching "point1", only one is removed. The
        optional "node" parameter is used for checking the identity, once the
        removeal candidate is decided."""

        # Recursion has reached an empty leaf node, nothing here to delete
        if not self:
            return

        # Recursion has reached the node to be deleted
        if self.should_remove(point1, node):
            return self._remove(point1)

        # Remove direct subnode
        if self.left and self.left.should_remove(point1, node):
            self.left = self.left._remove(point1)

        elif self.right and self.right.should_remove(point1, node):
            self.right = self.right._remove(point1)

        # Recurse to subtrees
        if point1[self.axis] <= self.data[self.axis]:
            if self.left:
                self.left = self.left.remove(point1, node)

        if point1[self.axis] >= self.data[self.axis]:
            if self.right:
                self.right = self.right.remove(point1, node)

        return self


    @require_axis
    def _remove(self, point1):
        # we have reached the node to be deleted here

        # deleting a leaf node is trivial
        if self.is_leaf:
            self.data = None
            return self

        # we have to delete a non-leaf node here

        # find a replacement for the node (will be the new subtree-root)
        root, max_p = self.find_replacement()

        # self and root swap positions
        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left, root.right = tmp_l if tmp_l is not root else self, tmp_r if tmp_r is not root else self
        self.axis, root.axis = root.axis, self.axis

        # Special-case if we have not chosen a direct child as the replacement
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point1, self)

        else:
            root.remove(point1, self)

        return root


    @property
    def is_balanced(self):
        """ Returns True if the (sub)tree is balanced

        The tree is balanced if the heights of both subtrees differ at most by
        1 """

        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0

        if abs(left_height - right_height) > 1:
            return False

        return all(c.is_balanced for c, _ in self.children)


    def rebalance(self):
        """
        Returns the (possibly new) root of the rebalanced tree
        """

        return create([x.data for x in self.inorder()])


    def axis_dist(self, point1, axis):
        """
        Squared distance at the given axis between
        the current Node and the given point1
        """
        return math.pow(self.data[axis] - point1[axis], 2)


    def dist(self, point1):
        """
        Squared distance between the current Node
        and the given point1
        """
        r = range(len(self.data))
        return sum([self.axis_dist(point1, i) for i in r])



    def get_path_to_best_matching_node(self, point1):
        "best matching node is the node before the leaf"
        return self.get_path_to_leaf(point1)[:-1]
        
    def get_path_to_leaf(self, point1):
        """ 
        Return the complete path, i.e. all nodes, to the closest leaf 
        """
        current = self
        assert current.active, "Root node not active, did you forget to activate some nodes?"
        
        prev = None
        
        # the nodes do not keep a reference to their parents
        parents = {current: None}
        passedNodes = []

        # go down the tree as we would for inserting
        while current and current.active:
            if point1[current.axis] < current.data[current.axis]:
                # left side
                parents[current.left] = current
                prev = current
                current = current.left
            else:
                # right side
                parents[current.right] = current
                prev = current
                current = current.right
            
 #           passedNodes.append(prev)
            passedNodes.append(current)
        if not prev:
            return []

        return passedNodes

       

    def search_knn(self, point1, k, dist=None):
        """ Return the k nearest neighbors of point1 and their distances

        point1 must be an actual point1, not a node.

        k is the number of results to return. The actual results can be less
        (if there aren't more nodes to return) or more in case of equal
        distances.

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any compareable type.

        The result is an ordered list of (node, distance) tuples.
        """

        prev = None
        current = self

        if dist is None:
            get_dist = lambda n: n.dist(point1)
        else:
            get_dist = lambda n: dist(n.data, point1)

        # the nodes do not keep a reference to their parents
        parents = {current: None}

        # go down the tree as we would for inserting
        while current:
            if point1[current.axis] < current.data[current.axis]:
                # left side
                parents[current.left] = current
                prev = current
                current = current.left
            else:
                # right side
                parents[current.right] = current
                prev = current
                current = current.right

        if not prev:
            return []

        examined = set()
        results = {}

        # Go up the tree, looking for better solutions
        current = prev
        while current:
            # search node and update results
            current._search_node(point1, k, results, examined, get_dist)
            current = parents[current]

        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key=BY_VALUE)


    def _search_node(self, point1, k, results, examined, get_dist):
        examined.add(self)

        # get current best
        if not results:
            bestNode = None
            bestDist = float('inf')

        else:
            bestNode, bestDist = sorted(results.items(), key=lambda n_d: n_d[1], reverse=True)[0]

        nodesChanged = False

        # If the current node is closer than the current best, then it
        # becomes the current best.
        nodeDist = get_dist(self)
        if nodeDist < bestDist:
            if len(results) == k and bestNode:
                results.pop(bestNode)

            results[self] = nodeDist
            nodesChanged = True

        # if we're equal to the current best, add it, regardless of k
        elif nodeDist == bestDist:
            results[self] = nodeDist
            nodesChanged = True

        # if we don't have k results yet, add it anyway
        elif len(results) < k:
            results[self] = nodeDist
            nodesChanged = True

        # get new best only if nodes have changed
        if nodesChanged:
            bestNode, bestDist = next(iter(
                sorted(results.items(), key=lambda n: n[1], reverse=True)
            ))

        # Check whether there could be any points on the other side of the
        # splitting plane that are closer to the search point1 than the current
        # best.
        for child, pos in self.children:
            if child in examined:
                continue

            examined.add(child)
            compare, combine = COMPARE_CHILD[pos]

            # Since the hyperplanes are all axis-aligned this is implemented
            # as a simple comparison to see whether the difference between the
            # splitting coordinate of the search point1 and current node is less
            # than the distance (overall coordinates) from the search point1 to
            # the current best.
            nodePoint = self.data[self.axis]
            pointPlusDist = combine(point1[self.axis], bestDist)
            lineIntersects = compare(pointPlusDist, nodePoint)

            # If the hypersphere crosses the plane, there could be nearer
            # points on the other side of the plane, so the algorithm must move
            # down the other branch of the tree from the current node looking
            # for closer points, following the same recursive process as the
            # entire search.
            if lineIntersects:
                child._search_node(point1, k, results, examined, get_dist)


    @require_axis
    def search_nn(self, point1, dist=None):
        """
        Search the nearest node of the given point1

        point1 must be an actual point1, not a node. The nearest node to the
        point1 is returned. If a location of an actual node is used, the Node
        with this location will be returned (not its neighbor).

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any compareable type.

        The result is a (node, distance) tuple.
        """

        return next(iter(self.search_knn(point1, 1, dist)), None)


    @require_axis
    def search_nn_dist(self, point1, distance, best=None):
        """
        Search the n nearest nodes of the given point1 which are within given
        distance

        point1 must be a location, not a node. A list containing the n nearest
        nodes to the point1 within the distance will be returned.
        """

        if best is None:
            best = []

        # consider the current node
        if self.dist(point1) < distance:
            best.append(self)

        # sort the children, nearer one first (is this really necessairy?)
        children = sorted(self.children, key=lambda c_p1: c_p1[0].dist(point1))

        for child, p in children:
            # check if child node needs to be recursed
            if self.axis_dist(point1, self.axis) < math.pow(distance, 2):
                child.search_nn_dist(point1, distance, best)

        return best


    @require_axis
    def is_valid(self):
        """ Checks recursively if the tree is valid

        It is valid if each node splits correctly """

        if not self:
            return True

        if self.left and self.data[self.axis] < self.left.data[self.axis]:
            return False

        if self.right and self.data[self.axis] > self.right.data[self.axis]:
            return False

        return all(c.is_valid() for c, _ in self.children) or self.is_leaf


    def extreme_child(self, sel_func, axis):
        """ Returns a child of the subtree and its parent

        The child is selected by sel_func which is either min or max
        (or a different function with similar semantics). """

        max_key = lambda child_parent: child_parent[0].data[axis]


        # we don't know our parent, so we include None
        me = [(self, None)] if self else []

        child_max = [c.extreme_child(sel_func, axis) for c, _ in self.children]
        # insert self for unknown parents
        child_max = [(c, p if p is not None else self) for c, p in child_max]

        candidates =  me + child_max

        if not candidates:
            return None, None

        return sel_func(candidates, key=max_key)
                                   
                    
    def _plot(self, x_min, y_min, x_max, y_max, plt, mark_labels=[]):
        plt.text(self.data[0], self.data[1], self.label, size=8)
        
        if self.label in mark_labels:
            plt.plot([x_min, x_max] , [y_min, y_max])
        
        if (self.left is not None and self.left.active) or (self.right is not None and self.right.active):        
            if self.axis == 0:
                plt.plot([self.data[self.axis], self.data[self.axis]] , [y_min, y_max ])
            
                if self.left:
                    #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
                    self.left._plot(x_min, y_min, self.data[self.axis],  y_max, plt, mark_labels)
            
                if self.right:
                    #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
                    self.right._plot(self.data[0], y_min, x_max, y_max, plt, mark_labels)
                
            elif self.axis == 1:
                plt.plot([x_min, x_max] , [self.data[self.axis], self.data[self.axis] ])
                if self.left:
                    #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
                    self.left._plot(x_min, y_min, x_max,  self.data[self.axis], plt, mark_labels)
            
                if self.right:
                    #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
                    self.right._plot(x_min, self.data[1], x_max, y_max, plt, mark_labels)
    
    
    def _plotV(self, x_min, y_min, x_max, y_max, ax, norm, V=numpy.empty(0)):
        plt_kd.text(self.data[0], self.data[1], "%0.2f" % V[self.label], size=8)
        color = cm.ocean( norm( float( V[self.label] ) ) ) 
        self._plotRect(x_min, y_min, x_max, y_max, ax, color)
        
        if (self.left and self.left.active) or (self.right and self.right.active):        
            if self.axis == 0:                
            
                if self.left:
                    #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
                    self.left._plotV(x_min, y_min, self.data[self.axis],  y_max, ax, norm, V)
            
                if self.right:
                    #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
                    self.right._plotV(self.data[0], y_min, x_max, y_max, ax, norm, V)
                
            elif self.axis == 1:
                #self._plotRect([x_min, x_max] , [self.data[self.axis], self.data[self.axis] ], colors[self.label])
                if self.left:
                    #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
                    self.left._plotV(x_min, y_min, x_max,  self.data[self.axis], ax, norm, V)
            
                if self.right:
                    #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
                    self.right._plotV(x_min, self.data[1], x_max, y_max, ax, norm, V) 
                    
                    
#     def _plotQ(self, x_min, y_min, x_max, y_max, ax, norm, Q=numpy.empty((0,0)) ):
#         if (self.left and self.left.active) or (self.right and self.right.active):        
#             if self.axis == 0:                
#                 if self.left:
#                     #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
#                     self.left._plotQ(x_min, y_min, self.data[self.axis],  y_max, ax, norm, Q)
#             
#                 if self.right:
#                     #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
#                     self.right._plotQ(self.data[0], y_min, x_max, y_max, ax, norm, Q)
#                 
#             elif self.axis == 1:
#                 #self._plotRect([x_min, x_max] , [self.data[self.axis], self.data[self.axis] ], colors[self.label])
#                 if self.left:
#                     #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
#                     self.left._plotQ(x_min, y_min, x_max,  self.data[self.axis], ax, norm, Q)
#             
#                 if self.right:
#                     #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
#                     self.right._plotQ(x_min, self.data[1], x_max, y_max, ax, norm, Q)
#             else:
#                 assert False, "Cannot plot this axis"
#         else:
#             colors = [cm.ocean(norm( float(q) )) for q in Q[self.label]] 
#             self._plotActionValues(x_min, y_min, x_max, y_max, ax, colors)                         

    def _plotQ(self, x_min, y_min, x_max, y_max, ax, norm, Q=numpy.empty((0,0)) ):
        if (self.left is not None and self.left.active) or (self.right is not None and self.right.active):        
            if self.axis == 0:                
                if self.left is not None:
                    #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
                    self.left._plotQ(x_min, y_min, self.data[self.axis],  y_max, ax, norm, Q)
            
                if self.right is not None:
                    #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
                    self.right._plotQ(self.data[0], y_min, x_max, y_max, ax, norm, Q)
                
            elif self.axis == 1:
                #self._plotRect([x_min, x_max] , [self.data[self.axis], self.data[self.axis] ], colors[self.label])
                if self.left is not None:
                    #print ("   call plot left:", x_min," ", y_min," ", self.data[1]," ",  y_max)
                    self.left._plotQ(x_min, y_min, x_max,  self.data[self.axis], ax, norm, Q)
            
                if self.right is not None:
                    #print ("   call plot right:", x_min, " ",self.data[1], " ",x_max, " ",y_max)
                    self.right._plotQ(x_min, self.data[1], x_max, y_max, ax, norm, Q)
            else:
                pass
#                assert False, "Cannot plot this axis"
        else:
            assert self.data is None, "At a leaf data is None. There is no further split."
            colors = [cm.ocean(norm( float(q) )) for q in Q[self.label]] 
            self._plotActionValues(x_min, y_min, x_max, y_max, ax, colors)       
                    
    
    def _plotActionValues(self, x_min, y_min, x_max, y_max, ax, color):
        x_top = (x_min + x_max)/2.0
        y_top = (y_min + y_max)/2.0
        
        #Assumption:
        # Action   Direction
        # 0        left
        # 1        right
        # 2        up
        # 3        down
        triangle_coord = [[x_min, y_min, x_min, y_max], [x_max, y_max, x_max, y_min], [x_max, y_min, x_min, y_min], [x_min, y_max, x_max, y_max]] 
                 
        assert len(triangle_coord)==len(color), "only four actions/colors supported"
        for col, coord in zip(color, triangle_coord):
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
#             x_text = 
#             y_text = (coord[1] + coord[3])
            
#            ax.text(self.data[0], self.data[1], "%0.2f" % V[self.label], size=8)  
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor=col, lw=2)
            ax.add_patch(patch) 
    
    
    def _plotRect(self, x_min, y_min, x_max, y_max, ax, color):
        verts = [
            (x_min, y_min), # left, bottom
            (x_min, y_max), # left, top
            (x_max, y_max), # right, top
            (x_max, y_min), # right, bottom
            (x_min, y_min), # ignored
            ]
         
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY,
                 ]
         
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color, lw=2)
        ax.add_patch(patch)
    
    
    
  
def update2DPlot(tree, min_coord=[0, 0], max_coord=[1, 1], mark_labels=[], plt=None, path_savefig=None):
    '''plots the splitting lines and the id of the splitting node'''
    #plt.figure(fig_kd.number)
    print ("update2DPlot")
    plt.ion()
    plt.clf()
    plt.xlim([min_coord[0], max_coord[0]])
    plt.ylim([min_coord[1], max_coord[1]])
    tree._plot(min_coord[0], min_coord[1], max_coord[0], max_coord[1], plt, mark_labels)
    plt.draw() 
    
    if path_savefig is not None:
        currentDateAndTime = str(datetime.datetime.now()).replace(".", ":") #plt.path_savefig interprets . as file extension
        plt_kd.savefig(path_savefig + currentDateAndTime)


def plot2D(tree, min_coord=[0, 0], max_coord=[1, 1], plt=None, mark_labels=[]):
    '''plots the splitting lines and the id of the splitting node'''       
    plt.xlim([min_coord[0], max_coord[0]])
    plt.ylim([min_coord[1], max_coord[1]])
    tree._plot(min_coord[0], min_coord[1], max_coord[0], max_coord[1], plt, mark_labels)
    plt.show()  



def plotQ2D(tree, min_coord=[0, 0], max_coord=[1, 1], Values=numpy.empty((0,0)), plt=None, plot="V", path_savefig=None):
    '''plots the splitting lines and the id of the splitting node'''
    #assert sum(1 for _ in level_order(tree, include_all=True))==len(Values), "Number of Q-Values does not match number of nodes" #Does not work, program freezes
    plt.ion()
    plt.clf()
    fig_v2d= plt.gcf()
    ax1_v2d = plt.gcf().gca()
#    plt_kd.figure(fig_v2d.number)

    ax1_v2d.set_xlim(min_coord[0],max_coord[0])
    ax1_v2d.set_ylim(min_coord[1],max_coord[1])
    
    norm = colors.Normalize(Values.min(), Values.max())

    if plot=="V":   #V
        tree._plotV(min_coord[0], min_coord[1], max_coord[0], max_coord[1], ax1_v2d, norm, V=Values)
    elif plot=="Q":               #Q
        tree._plotQ(min_coord[0], min_coord[1], max_coord[0], max_coord[1], ax1_v2d, norm, Q=Values)
    else:
        raise Exception("No valid Value function to display chosen")
    
    #visualize(tree)
    
    cax = ax1_v2d.imshow(Values, interpolation='nearest', cmap=cm.ocean)
    number_bins_colorbar = 20
    ticks = numpy.linspace(Values.min(), Values.max(), number_bins_colorbar)
    cbar = fig_v2d.colorbar(cax, ticks=[ticks], orientation='vertical')
    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    
    if path_savefig is not None:
        currentDateAndTime = str(datetime.datetime.now()).replace(".", ":") #plt.path_savefig interprets . as file extension
        plt.savefig(path_savefig + currentDateAndTime)      
    
    plt.draw()
    


    
#     
# def plotV2D(tree, min_coord=[0, 0], max_coord=[1, 1], V=numpy.empty(0), path_savefig=None):
#     '''plots the splitting lines and the id of the splitting node'''
#     plt_kd.ioff()
#     fig_v2d= plt_kd.figure()
#     ax1_v2d = fig_v2d.add_subplot(111)
#     plt_kd.figure(fig_v2d.number)
# 
#     ax1_v2d.set_xlim(min_coord[0],max_coord[0])
#     ax1_v2d.set_ylim(min_coord[1],max_coord[1])
#     
#     norm = colors.Normalize(V.min(), V.max())
#     v_color = [cm.ocean(norm( float(v) )) for v in V]
#     
#     
#     if path_savefig is not None:
#         currentDateAndTime = str(datetime.datetime.now()).replace(".", ":") #plt.path_savefig interprets . as file extension
#         plt_kd.savefig(path_savefig + currentDateAndTime)      
#     
#     plt_kd.show()
    


def createNewTree(point_list=None, dimensions=None, axis=0, sel_axis=None):
    global id_label
    id_label = 0
    return create(point_list, dimensions, axis, sel_axis)
    
    

def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """ Creates a kd-tree from a list of points

    All points in the list must be of the same dimensionality.

    If no point_list is given, an empty tree is created. The number of
    dimensions has to be given instead.

    If both a point_list and dimensions are given, the numbers must agree.

    Axis is the axis on which the root-node should split.

    sel_axis(axis) is used when creating subnodes of a node. It receives the
    axis of the parent node and returns the axis of the child node. """
    
#    assert sel_axis, "sel_axis is None. Only temporary, remove when adding nodes with adaptable splitting axes works"

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    # by default cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis+1) % dimensions)

    if not point_list:
        node = KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)
        return node

    # Sort point1 list and choose median as pivot element
    point_list.sort(key=lambda point1: point1[axis])
    median = len(point_list) // 2
    
    loc   = point_list[median]
    left  = create(point_list[:median], dimensions, sel_axis(axis), sel_axis = sel_axis)
    right = create(point_list[median + 1:], dimensions, sel_axis(axis), sel_axis = sel_axis)

    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)

def check_dimensionality(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')

    return dimensions


def getNode(tree, label):
    for n in level_order(tree):
        if n.label == label:
            return n
    

def level_order(tree, include_all=False):
    """ Returns an iterator over the tree in level-order

    If include_all is set to True, empty parts of the tree are filled
    with dummy entries and the iterator becomes infinite. """

    q = deque()
    q.append(tree)
    while q:
        node = q.popleft()
        yield node

        if node.left is not None:
            q.append(node.left)
        elif include_all:
            q.append(node.__class__(label=-1)) 
        

        if node.right is not None:
            q.append(node.right)
        elif include_all:
            q.append(node.__class__(label=-1)) 

def save(tree, filename):
    pickle.dump( tree, open( filename + "_kdtree.pkl", "wb" ) )
    
def load(filename):
    return pickle.load( open( filename+ "_kdtree.pkl", "rb" ) )

def visualize(tree, max_level=100, node_width=10, left_padding=5):
    """ Prints the tree to stdout """

    #height = min(max_level, tree.height()-1)
    height = min(max_level, tree.height())
    max_width = pow(2, height)

    per_level = 1
    in_level  = 0
    level     = 0
    
    edge_list=[]

    for node in level_order(tree, include_all=True):

        if in_level == 0:
            print()
            print()
            print(' '*left_padding, end=' ')

        width = int(max_width*node_width/per_level)

        #node_str = ("id: " + str(node.label) + " " + str(node.data) if node else '').center(width)
        node_str = ("{id:" + str(node.label) + ", ax:" + str(node.axis) + " " + str(node.data) + " " + str(node.active) + "}" ).center(width)
        
        if node is not None and node.left is not None: 
            edge_list.append((node.label,  node.left.label))
        if node is not None and node.right is not None:   
            edge_list.append((node.label, node.right.label))
            
        print(node_str, end=' ')

        in_level += 1

        if in_level == per_level:
            in_level   = 0
            per_level *= 2
            level     += 1

        if level > height:
            break

    print()
    print()
    return edge_list
