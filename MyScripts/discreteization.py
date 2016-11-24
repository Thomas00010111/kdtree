#import kdtree
import kdtree
import numpy
import util
import matplotlib.pyplot as plt


class discreteization():
    def __init__(self):
        nodes = []
        points = numpy.array([[0.0, 0.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 1.0]])
        util.splitN(points, 0, 0, 4, nodes)
        
        #print "nodes:", nodes  
        print "Number of node: ", len(nodes)
        self.tree = kdtree.createNewTree(nodes)
        
        util.activate(self.tree, 3)
        
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        


    def update2DPlot(self, min_coord=[0, 0], max_coord=[1, 1], mark_labels=[], path_savefig=None):
        plt.figure(self.fig.number)
        kdtree.update2DPlot(self.tree, min_coord=min_coord, max_coord=max_coord,  mark_labels= mark_labels, plt=plt, path_savefig=path_savefig)
        
    def plotQ2D(self, min_coord=[0, 0], max_coord=[1, 1], Values=numpy.empty((0,0)), plot="V", path_savefig=None):
        plt.figure(self.fig.number)
        kdtree.plotQ2D(self.tree, min_coord=min_coord, max_coord=max_coord, plt=plt, Values=Values, path_savefig=path_savefig)
