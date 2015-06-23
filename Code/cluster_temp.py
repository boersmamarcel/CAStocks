# source: http://dragly.org/2013/03/25/working-with-percolation-clusters-in-python/

from scipy.ndimage import measurements
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

L = 100 # array size (square)
r = np.random.random((L,L)) # get random

# necessary for example
p = 0.4 # threshhold for being deemed 'active'
z = r < p # matrix of true/false values
 
# plot random numbers
plt.subplot(221)
plt.imshow(z, origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Matrix")
plt.show()

# plot cluster labels
plt.subplot(222)
lw, num = measurements.label(z) # lw: matrix with cluster numbers, num: total number of clusters
plt.imshow(lw, origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Labeled clusters")
plt.show()

# plot shuffled cluster labels for better color distinction
plt.subplot(223)
b = np.arange(num + 1) # array of cluster labels
np.random.shuffle(b) # shuffle cluster labels
shuffledLw = b[lw] # implement shuffled labels
plt.imshow(shuffledLw, origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Shuffled label clusters")
plt.show()

# plot area cluster
plt.subplot(224)
area = measurements.sum(z, lw, index=np.arange(num + 1)) # get array of cluster size
areaImg = area[lw] # swap cluster index with cluster area
im3 = plt.imshow(areaImg, origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Clusters by area")
plt.show()
