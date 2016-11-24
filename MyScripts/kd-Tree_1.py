import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
 
def synthesizeData(size,dim,scale=1.0):
    data = np.empty((size, dim))
    for d in range(dim):
        data[:,d] = scale*np.random.random(size)
    return data
 
def main():
    # settings
    scale = 100
    size = 1000
    dim = 2
 
    # build kd-tree
    data = synthesizeData(size, dim, scale)
    tree = spatial.KDTree(data)
 
    # search
    for numQuery in range(1,5):
        print 'search ', numQuery, 'queries'
        query = synthesizeData(numQuery, dim, scale)
        knnIndex = tree.query_ball_point(query, 15.0)
        plt.plot(data[:,0], data[:,1], 'ro')
        for k in range(numQuery):
            print '    plot ', k, '-th data'
            if k == 0:
                plt.plot(data[knnIndex[k],0], data[knnIndex[k],1], 'bo')
            if k == 1:
                plt.plot(data[knnIndex[k],0], data[knnIndex[k],1], 'go')
            if k == 2:
                plt.plot(data[knnIndex[k],0], data[knnIndex[k],1], 'yo')
            if k == 3:
                plt.plot(data[knnIndex[k],0], data[knnIndex[k],1], 'ko')
    
    plt.show()
 
if __name__ == '__main__':
    main()