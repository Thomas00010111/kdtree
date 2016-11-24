import numpy
import itertools


'''Equally split leave nodes in tree'''
    
def splitN(points, level, axis, maxlevel, nodes):
    if level <= maxlevel:
        numberOfPoints = points.shape[0];
        nodes.append(numpy.array(sum(points)/numberOfPoints).tolist())
        
        #put larger points in an array and smaller points in an array
        right_branch=numpy.empty(points.shape)
        left_branch=numpy.empty(points.shape)
        avg_axis=sum(points[:,axis])/numberOfPoints
       
        index_left, index_right = 0, 0
        for p in points:
            if p[axis] <=  avg_axis:
                left_branch[index_left]=p
                index_left+=1
            else:
                right_branch[index_right]=p
                index_right+=1
        
        assert len(left_branch) == len(right_branch), "Number of points has to be equal in both lists"        
        #generate new border
        newPoints=[]
        for p in range(0, index_right):
            temp = right_branch[p].copy()
            temp[axis] = avg_axis
            newPoints.append(temp) 
        
        for n in newPoints:      
            left_branch[index_left] =n
            index_left+=1
            right_branch[index_right] =n
            index_right+=1
        
        axis=(axis+1)%points.shape[1]
        splitN(left_branch, level+1, axis, maxlevel, nodes )
        splitN(right_branch, level+1, axis, maxlevel, nodes )

def activate(node, maxlevel, level=0):
    if level <= maxlevel:
        assert node is not None, "No node to activate"
        node.active=True
        activate(node.left, maxlevel, level+1,)
        activate(node.right, maxlevel, level+1)

def generate_sequence_of_points(dimension, levels):
    nodes = []
    sequence = ["".join(seq) for seq in itertools.product("01", repeat=dimension)]
    points_temp= numpy.array([list(s) for s in sequence])
    points = numpy.array([map(float, f) for f in points_temp])
    splitN(points, 0, 0, levels, nodes)
    return nodes