import kdtree
import numpy



# A kd-tree can contain different kinds of points, for example tuples
point1 = (1, 1)

# Lists can also be used as points
point2 = (1.5, 1)
point3 = (0.5, 1)
point4 = (0.5, 1.5)
point5 = (0.5, 0.5)

# Other objects that support indexing can be used, too
# import collections
# Point = collections.namedtuple('Point', 'x y z')
# point3 = Point(5, 3, 2)

# A tree is created from a list of points
tree = kdtree.create([point1, point2, point3, point4, point5])

kdtree.visualize(tree)

#print list(kdtree.level_order(tree))

print tree.get_path_to_best_matching_node((0.9,1))
print tree.get_path_to_best_matching_node((1.1,1))
print tree.get_path_to_best_matching_node((1.0,1.1))
print tree.get_path_to_best_matching_node((0.999,1.1))


tree.add((0.9, 1.1))
tree.add((1.2, 1.0))

kdtree.visualize(tree)

print tree.get_path_to_best_matching_node((0.9,1))
print tree.get_path_to_best_matching_node((1.1,1))
print tree.get_path_to_best_matching_node((1.0,1.1))
print tree.get_path_to_best_matching_node((0.999,1.1))
print tree.get_path_to_best_matching_node((0.499,1.1))
print tree.get_path_to_best_matching_node((0.499,1.6))

tree = tree.rebalance()

kdtree.visualize(tree)