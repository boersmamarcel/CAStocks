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
import matplotlib.mlab as mlab


cuda.select_device(0) #select videocard

w = 20
h = 20

initProb = 0.05
pim = 0.0 # probability of being an imitator

#generate random traders
A = np.array(np.random.choice([0, 1], p=[1-initProb, initProb], size=w*h, replace=True).reshape(h,w), dtype=np.int32)
B = np.empty_like(A)
C = np.empty_like(B)

for i in range(h):
    for j in range(w):
        C[i,j] = np.random.choice([1,2],p=[1-pim, pim]) if A[i,j] == 1 else 0


print np.where(C == 1)[0].size
print np.where(C == 2)[0].size
print np.where(C != 0)[0].size
print np.where(A == 1)[0].size

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

@cuda.jit(restype = f4, argtypes=[int32, f4, f4, f4[:], f4[:], f8[:], f8], device=True)
def localIRule(k, A, a, clusterSize, nClustOnes, xi, eta):
# normalization constant
    I = 1./clusterSize[k-1]
    
    # sum active traders +1 and -1 states
    A = 1.8
    a = 2*A
    
    n1 = nClustOnes[k-1]
    n2 = clusterSize[k-1] - nClustOnes[k-1]

    c = 0.0
    c += ( A*(xi[k-1]*2-1)*n1 + a*(eta*2*n1 -n1) )*1 # positive spins 
    c += ( A*(xi[k-1]*2-1)*n2 + a*(eta*2*n2 -n2) )*-1 # negative spins 
        
    return I*c #+ self.calch() 




@cuda.jit(restype = f4, argtypes=[int32, int32, f4, f4, f4, f4, f4[:], f4[:], f8[:], f8], device=True)
def localPRule(k, fundState, price, funPrice, A, a, clusterSize, nClustOnes, xi, eta):
    if fundState == 1:
        #fundamentalist, so determine state spin with fundamental price
        diff = price - funPrice 
        #if the stock is overvalued then you want to sell (-1) else buy (1)
        if diff <= 0:
            #over valued
            return 1.0
        else:
            return 0.0
    elif fundState == 2:
        #immitator, so align spin
        return 1./(1+math.exp(-2*localIRule(k, A, a, clusterSize, nClustOnes, xi, eta)))
    else:
        return 0


@cuda.jit(argtypes=[f4, f4, int32[:,:], int32[:,:], int32[:,:], int32[:,:], f4[:], int32, f4[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f4], target='gpu')
def stateUpdate(price, fundaPrice, currentGrid, nextGrid, fundGrid, cluster, clusterSize, nClust, clusterOnes, enterP, activateP, choiceP, diffuseP, fundP, xis, eta, p_im):
    
    x,y = cuda.grid(2)
    gw,gh = currentGrid.shape

    #settings
    pe = 0.0001
    pd = 0.05
    ph = 0.0485/1.5
    
    A = 1.8
    a = 2*A
#    h = 0

    if x < gw and y < gh: 
        cellState = currentGrid[x,y]
        fundState = fundGrid[x,y]
        width, height = currentGrid.shape

        if cellState == 0:
            # this is the enter probability, where a person has a certain probability of entering the market
            enter = enterP[x*width + y] 


            #activation probability
            if enter < pe: 
                #will it be a fundamentalist or immitator at entering state, stays the same for the rest of the time
                fundState = 1 if fundP[x*width + y] < 1-p_im else 2
                if fundState == 2:
                    cellState = 1 if choiceP[x*width + y] < 0.5 else -1
                else:
                    cellState = 1 if price - fundaPrice <= 0 else -1
                

            else:
                #activate by neighbors
                vals = 0 #majority vote
                neighbours = 0
                if abs(currentGrid[x-1,y]) == 1:
                    neighbours += 1
                    vals += currentGrid[x-1,y]
                if abs(currentGrid[x,y-1]) == 1:
                    neighbours += 1
                    vals += currentGrid[x,y-1]
                if abs(currentGrid[(x+1)%width,y]) == 1:
                    neighbours += 1
                    vals += currentGrid[(x+1)%width,y]
                if abs(currentGrid[x,(y+1)%height]) == 1:
                    neighbours += 1
                    vals += currentGrid[x,(y+1)%height]
        
                activated = (1-(1-ph)**neighbours)
                if activateP[x*width + y] < activated:
                    fundState = 1 if fundP[x*width + y] < 1-p_im else 2
                    if fundState == 2:
                        cellState = 1 if choiceP[x*width + y] < 0.5 else -1
                    else:
                        cellState = 1 if price - fundaPrice <= 0 else -1
                        
        elif abs(cellState) == 1:
            
            #If the price is around the true value, not much is expected so people leave the market

            if diffuseP[x*width + y] < pd:
                cellState = 0
                fundState = 0 #neither fundamentalist nor immitator
            else:
                k = cluster[x,y] #which cluster

                # choiceP double used for eta and pk test
                # also same eta is used for each cluster interaction!
                pk = localPRule(k, fundState, price, fundaPrice, A, a, clusterSize, clusterOnes, xis, eta[x*width + y])

                if choiceP[x*width + y] < pk:
                    cellState = 1
                else:
                    cellState = -1
                    
        nextGrid[x,y] = cellState
        fundGrid[x,y] = fundState

def updatePrice(price, clusterSize, nClustOnes):
    # what is beta?????
    x = 1.0/(w*h*w*h)
    # matrix form of summation
    vals = np.sum( np.multiply(clusterSize, nClustOnes-(clusterSize-nClustOnes) ) )
           
    x *= vals
    price += price*x # update price
    
    if price < 0:
        price = 0 # lower bound

    return x, price
    
#upload memory to gpu
bpg = 50
tpb = 32

nView = 3
steps = 3

initialPrice = 100

stream = cuda.stream() #initialize memory stream

# instantiate a cuRAND PRNG
prng = curand.PRNG(curand.PRNG.MRG32K3A, stream=stream)

paths = 1

pricePath = []

fundPrice = 500.

for j in range(paths):
    print "Generating path: %s" % j
    # plotting lists
    LogReturns, nLogReturns = [0], [0] # log returns, normalized log returns
    xchange, xcorrelation = [], [] # change in price P, and autocorrelation
    activeTraders = [] # number of active traders
    prices = [initialPrice]
    price = initialPrice

    for i in range(steps):

        # Allocate device side array
        enterProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
        activateProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
        choiceProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
        diffuseProbs = cuda.device_array(w*h, dtype=np.double, stream=stream)
        fundamentaListProbs = cuda.device_array(w*h, dtype=np.double, stream = stream)

        #calculate cluster info
        cluster, clusterSize, nClust, nClustOnes = calcCluster(A) # get cluster info

        xis = cuda.device_array(nClust, dtype=np.double, stream = stream)
        eta = cuda.device_array(w*h, dtype=np.double, stream = stream) # 1 -> w*h*w*h, how to index???


        with stream.auto_synchronize():
            dA = cuda.to_device(A, stream) #upldate grid
            dB = cuda.to_device(B, stream) #upload new locatoin
            dC = cuda.to_device(C, stream) #fundamentalist/immitators grid

            dCluster = cuda.to_device(cluster, stream) #upload cluster grid to GPU
            dClusterSize = cuda.to_device(clusterSize, stream) #upload cluster size
            dnClustOnes  = cuda.to_device(nClustOnes, stream) #upload ones per cluster


            prng.uniform(enterProbs) #generate first random number
            prng.uniform(activateProbs) #generate second random number
            prng.uniform(diffuseProbs) #generate third random number
            prng.uniform(choiceProbs) #generate extra random number

            prng.uniform(fundamentaListProbs)
            
            prng.uniform(xis)
            prng.uniform(eta)

            stateUpdate[(bpg, bpg), (tpb, tpb), stream](price, fundPrice, dA, dB, dC, dCluster, dClusterSize, nClust, dnClustOnes, enterProbs, activateProbs, choiceProbs, diffuseProbs, fundamentaListProbs, xis, eta, pim)

            #get GPU memory results
            dB.to_host(stream)
            dC.to_host(stream)
            
            

        #set new grid as current grid and repeat
        A = B.copy()
        print np.where(C==1)[0].size, np.where(C==2)[0].size, np.where(np.absolute(B)==1)[0].size
        print C
        print B
        D = np.subtract(B,C)
        print D, np.where(D!=0)[0].size

        x, price = updatePrice(price, clusterSize, nClustOnes)
        prices.append(price)
        xchange.append(x)
        activeTraders.append(np.sum(np.absolute(A))/(1.0*w*h))
        
        # update returns
        LogReturns = np.append(LogReturns, np.log(prices[i]) - np.log(prices[i-1]) )
        nLogReturns = np.append(nLogReturns, (LogReturns[i] - np.mean(LogReturns))/np.std(LogReturns) )

        if i % int(steps/nView) == 0:
            cmap = mpl.colors.ListedColormap(['red','white','green'])
            bounds=[-1.1,-.1,.1,1.1]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            im = plt.imshow(B.astype(int),interpolation='nearest', cmap = cmap,norm=norm)

            plt.show()
    
        pricePath.append(prices)
        
    # make some plots
    print "fraction active traders"
    plt.figure()
    plt.plot(activeTraders)
    plt.show()
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    hist, bin_edges = np.histogram(clusterSize, bins=20)
#    print "cluster size distribution", hist
#    ax.plot(hist)
#    ax.set_xscale('log')
#    ax.set_yscale('log')
#    plt.show()
    
    print "normalized log returns"
    plt.figure()
    plt.plot(nLogReturns)
    plt.show()

    print "normalized log retrun distribution"
    plt.figure()
    mu, sigma = np.mean(nLogReturns), np.std(nLogReturns)
    xmin, xmax = min(np.amin(nLogReturns),-7) , max(np.amax(nLogReturns), 7)
    x = np.linspace(xmin, xmax, 100)
    p1, = plt.plot(x,mlab.normpdf(x,mu,sigma), label='normal distribution')
    
    hist, bin_edges = np.histogram(nLogReturns, bins=20, normed=True)
    bin_means = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(hist))]
    p2 = plt.scatter(bin_means, hist, marker='o', label='model')
    
#    hist, bin_edges = np.histogram(sp500_nlogReturns, bins=20, normed=True)
#    bin_means = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(hist))]
#    p3 = plt.scatter(bin_means, hist, marker='v', color='g', label='sp500')
    
    plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc=2)
    plt.yscale('log')
    plt.xlim([xmin, xmax])
    plt.xlabel('normalized logreturn values [-]')
    plt.ylabel('Probability [-]')
    plt.show()
    
    volatility = np.array([])
    for i in range(1,len(prices)): # compute volatility
        ki = 400 if 400 < i else i
        price_avg = np.mean(prices[i-ki:i+1])
        value = np.sum(np.absolute(prices[i-ki:i+1] - price_avg))/(1.0*ki*price_avg)
        volatility = np.append(volatility, value)
    
    v_clustering = np.array([])
    for lag in range(1,500): # array of correlation
        xcorrelation = np.append(xcorrelation,  np.sum(np.multiply(xchange[lag:],xchange[:-lag])))
        v_clustering = np.append(v_clustering,  np.sum(np.multiply(volatility[lag:],volatility[:-lag])))
    xcorrelation = xcorrelation/xcorrelation[0] # normalize to first entry
    v_clustering = v_clustering/v_clustering[0] # normalize to first entry
    
    plt.figure()
    p1, = plt.plot(xcorrelation, label='price clustering')
    p2, = plt.plot(v_clustering, label='volatility clustering')
#    p3, = plt.plot(sp500_price_clustering, label='sp500 price clustering')
#    p4, = plt.plot(sp500_volatility_clustering, label='sp500 volatility clustering')
    plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlabel('Lag [-]')
    plt.ylabel('Correlation coeficient [-]')
    plt.show() 
        
        


plt.figure()
for j in range(len(pricePath)):
    plt.plot(pricePath[j])
plt.show()

#print np.matrix(B).max()