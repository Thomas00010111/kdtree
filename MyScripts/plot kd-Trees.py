#import kdtree
import numpy
import time
import discreteization

ds = []
for d in range(2):
    ds.append(discreteization.discreteization())
    
    
for d in ds:            
    d.update2DPlot()
    

for d in ds:
    d.plotQ2D(Values = numpy.random.rand(31,1))

while(1):
    time.sleep(0.5)





