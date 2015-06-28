from numbapro import cuda
from numba import *
import numpy as np
import math
from timeit import default_timer as time
from numbapro.cudalib import curand
import math

from scipy.ndimage import measurements

import matplotlib.pyplot as plt
import matplotlib as mpl


cuda.select_device(0) #select videocard

w = 120
h = 30

initProb = 0.01

#generate random traders
A = np.array(np.random.choice([0, 1,-1], p=[1-initProb, initProb/2, initProb/2], size=w*h, replace=True).reshape(h,w), dtype=np.int32)
B = np.empty_like(A)

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

@cuda.jit(restype = f4, argtypes=[int32, f4[:], f4[:], f8[:], f8], device=True)
def localIRule(k, clusterSize, nClustOnes, xi, eta):
       # normalization constant
    I = 1./clusterSize[k-1]
    
    # sum active traders +1 and -1 states
    A = 1.8
    a = 2*1.8


    c = 0.0
    anA = A*xi[k-1] + a*eta
    for i in range(int(nClustOnes[k-1])): # positive spins
        c += anA*1
    for i in range(int(clusterSize[k-1] - nClustOnes[k-1])): # negative spins
        c += anA*-1
        
    return I*c #+ self.calch() 


@cuda.jit(restype = f4, argtypes=[int32, f4[:], f4[:], f8[:], f8], device=True)
def localPRule(k, clusterSize, nClustOnes, xi, eta):
	return 1./(1+math.exp(-2*localIRule(k, clusterSize, nClustOnes, xi, eta)))

@cuda.jit(restype = int32, argtypes=[int32[:,:], int32[:,:],f4[:], int32, f4[:], int32, int32, int32, f4, f4, f4, f4, f4, f4, f4, f4, f8[:], f8[:], f8[:], f8[:], f8[:]], device=True)
def cellUpdate(grid, cluster, clusterSize, nClust, clusterOnes, x, y, i,  pe, pd, ph, price,  A, a, h, beta, enterP, activateP, choiceP, diffuseP, xis):
	cellState = grid[x,y]

	width, height = grid.shape

	if cellState == 0:
		enter = activateP[x*width + y]

		#activation probability
		if enter < 0.5: 
			cellState = 1 if choiceP[x*width + y] < 0.5 else -1
		else:
			#activate by neighbors
			neighbours = 0
			if abs(grid[x-1,y]) == 1:
			    neighbours += 1
			if abs(grid[x,y-1]) == 1:
			    neighbours += 1
			if abs(grid[(x+1)%width,y]) == 1:
			    neighbours += 1
			if abs(grid[x,(y+1)%height]) == 1:
			    neighbours += 1

			activated = (1-(1-ph)**neighbours)
			if activateP[x*width + y] < activated:
			    cellState = 1 if choiceP[x*width + y] < 0.5 else -1

	elif abs(cellState) == 1:
			if diffuseP[x*width + y] < pd:
				cellState = 0
			else:
				k = cluster[x,y] #which cluster
				pk = localPRule(k, clusterSize, clusterOnes, xis, choiceP[x*width + y])

				if choiceP[x*width + y] < pk:
					cellState = 1
				else:
					cellState = -1


	return cellState 


@cuda.jit(argtypes=[int32[:,:], int32[:,:], int32[:,:], f4[:], int32, f4[:], f8[:], f8[:], f8[:], f8[:], f8[:]], target='gpu')
def stateUpdate(currentGrid, nextGrid, cluster, clusterSize, nClust, clusterOnes, enterP, activateP, choiceP, diffuseP, xis):
	x,y = cuda.grid(2)
	i 	= cuda.grid(1)
	gw,gh = currentGrid.shape

	#settings
	price = 100

	pe = 0.0001
	pd = 0.05
	ph = 0.0485/1.5
	
	A = 1.8
	a = 2*A
	h = 0
	beta = 0.000001


	if x < gw and y < gh: 
		nextGrid[x,y] = cellUpdate(currentGrid, cluster, clusterSize, nClust, clusterOnes, x, y, i, pe, pd, ph, price,  A, a, h, beta, enterP, activateP, choiceP, diffuseP, xis)


#upload memory to gpu
bpg = 50
tpb = 32

nView = 5
steps = 500

stream = cuda.stream() #initialize memory stream

# instantiate a cuRAND PRNG
prng = curand.PRNG(curand.PRNG.MRG32K3A, stream=stream)

for i in range(steps):

	# Allocate device side array
	enterProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
	activateProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
	choiceProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
	diffuseProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)

	#calculate cluster info
	cluster, clusterSize, nClust, nClustOnes = calcCluster(A) # get cluster info

	xis 		= cuda.device_array(nClust, dtype=np.double, stream = stream)


	with stream.auto_synchronize():
		dA = cuda.to_device(A, stream) #upldate grid
		dB = cuda.to_device(B, stream) #upload new locatoin
		dCluster = cuda.to_device(cluster, stream) #upload cluster grid to GPU
		dClusterSize = cuda.to_device(clusterSize, stream) #upload cluster size
		dnClustOnes  = cuda.to_device(nClustOnes, stream) #upload ones per cluster
		dxis 		= cuda.to_device(xis, stream)

		prng.uniform(enterProbs) #generate first random number
		prng.uniform(activateProbs) #generate second random number
		prng.uniform(choiceProbs) #generate second random number
		prng.uniform(diffuseProbs) #generate second random number
		prng.uniform(dxis)

		stateUpdate[(bpg, bpg), (tpb, tpb), stream](dA, dB, dCluster, dClusterSize, nClust, dnClustOnes, enterProbs, activateProbs, choiceProbs, diffuseProbs, dxis)

		#get GPU memory results
		dB.to_host(stream)

	if i % int(steps/nView) == 0:
		cmap = mpl.colors.ListedColormap(['red','white','green'])
		bounds=[-1.1,-.1,.1,1.1]
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

		im = plt.imshow(B.astype(int),interpolation='nearest',
		                    cmap = cmap,norm=norm)

		# plt.show()


	#set new grid as current grid and repeat
	A = B.copy()


print np.matrix(B).max()