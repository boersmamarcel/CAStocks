# optimized version
import numpy as np
from scipy.ndimage import measurements

from numbapro import vectorize, int32, cuda, void

def calcXi(n):
    # random variable xi_k
    # calculate xi list with predefined random variable per cluster
    return np.random.uniform(-1,1,n)

# calculate cluster info 
def calcCluster(grid):
    grid_abs = np.absolute(grid) # reduce field to active/inactive traders
    grid_abs = grid_abs == 1 # get field of True/False values
    
    # lw: matrix with cluster numbers, num: total number of clusters, area: matrix of cluster size 
    lw, num = measurements.label(grid_abs) 
    area = measurements.sum(grid_abs, lw, index=np.arange(1,num+1))  

    cluster_ones = np.zeros(num) # define empty array
    for i in range(1,num+1): # loop clusters
        cluster_ones[i-1] = (np.where(np.logical_and(lw==i,grid==1))[0]).size # get number of +1 states in cluster

    return lw, area, num, cluster_ones

@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'], target='parallel')
def doCellStep(a, b):
    return a + b

# simulation step function
def doStep(grid):
	nextGrid = grid.copy()
	cluster, clusterSize, nClust, nClustOnes = calcCluster(grid)

	wrange = np.arange(100)
	hrange = np.arange(100)

	indexTuples = [ [(i,j) for j in wrange] for i in hrange]

	doCellStep(wrange, hrange)










# set settings

p = 0.01 #initialize grid probability
pe = 0 #enter probability
pd = 0 #diffuse probability
ph = 0 #neighbor activation probability
price = 0 # initial stock price

x = 0

# some parameters
A = 1.8
a = 2*A
h = 0
beta = 0.000001

#area
width = 512
height = 128

# initialize grid
grid = np.random.binomial(1, p, width*height).reshape(width, height)

# initial cluster info
cluster, clusterSize, nClust, nClustOnes = calcCluster(grid)

# initialize xi
xi = calcXi(nClust)

doStep(grid)
