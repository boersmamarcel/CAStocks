import numpy as np
from scipy.ndimage import measurements

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.mlab as mlab

import pandas as pd
import scipy.stats as scs

class CAmodel(object):
    
    def __init__(self, width, height, steps, nView, initProb, initPrice, fundPrice, p_im, pe, pd, ph):
        
        # constants
        self.width = width
        self.height = height
        
        self.steps = steps
        self.nView = nView
        
        self.A = 1.8
        self.a = 2*self.A
        self.h = 0
        
        self.initProb = initProb
        self.p_im = p_im
        
        self.pe = pe
        self.pd = pd
        self.ph = ph
        
        self.initPrice = initPrice
        self.Price_List = np.array([initPrice])
        self.fundPrice = fundPrice
        
        # define grids
        # initialize in +1 spin state with probability initProb
        self.grid = (np.random.uniform(0,1,width*height).reshape(width,height) < initProb)*1
        self.gridNext = self.grid.copy()
        
        # set trader type: fundamentalist (1) or imitator (2)
        self.active_idx = np.where(self.grid != 0) # get active cells
        self.fundGrid = np.zeros((width, height)) # set empty grid
        self.fundGrid[self.active_idx] = np.random.choice([1,2], self.active_idx[0].size, p=[1-p_im, p_im])
        
        # plotting lists
        self.xChange = np.array([])
        
        self.logReturn = np.array([0])
        self.nlogReturn = np.array([0])
        
        self.activeTraders = np.array([])
    
    def calcCluster(self):
        
        grid_abs = np.absolute(self.grid) # reduce field to active/inactive traders
        grid_abs = grid_abs == 1 # get field of True/False values
            
        # lw: matrix with cluster numbers, num: total number of clusters, area: matrix of cluster size 
        lw, num = measurements.label(grid_abs) 
        area = measurements.sum(grid_abs, lw, index=np.arange(1,num+1))  
    
        # ones:  number of +1 states in cluster
        ones = np.zeros(num)
        for i in range(1,num+1): 
            ones[i-1] = (np.where(np.logical_and(lw==i,self.grid==1))[0]).size
    
        return lw, area, num, ones 
    
    def localIRule(self, k):
        
        I = 1.0/self.clusterSize[k-1]
        
        n1 = self.clusterOnes[k-1]
        n2 = self.clusterSize[k-1] - self.clusterOnes[k-1]
        
        xi = self.Xi_list[k-1] # cluster term (k)
        eta = np.random.uniform(-1,1) # interaction term (i,j)
        dzeta = np.random.uniform(-1,1) # external term (k,i)
    
        c = 0.0
        c += ( self.A*xi*n1 + self.a*eta*n1)*1 # positive spins 
        c += ( self.A*xi*n2 + self.a*eta*n2)*-1 # negative spins 
        
        return I*c + self.h*dzeta       
    
    def localPRule(self, k, fundState):
        
        diffPrice = self.Price - self.fundPrice
        
        if fundState == 1: # fundamentalist
            p = 1.0 if diffPrice <= 0 else 0.0
#            print 'fundamtalist detected'
        elif fundState == 2: # imitator
            p = 1.0/(1.0 - np.exp(-2*self.localIRule(k))) 
#            print 'imitator detected'
        else:
            p = 0.0
            print 'error in state'
        return p

    def updatePrice(self):
        
        beta = 1.0/(self.width*self.height)**2
        x = beta*np.sum( np.multiply(self.clusterSize, self.clusterOnes-(self.clusterSize-self.clusterOnes) ) )

        # remove cluster weight test
#        beta = 1.0/(self.width*self.height)
#        x = beta*np.sum(self.clusterOnes-(self.clusterSize-self.clusterOnes))


        self.Price += x*self.Price # update price
        
        if self.Price < 0:
            self.Price = 0 # lower bound
    
        return x
    
    def cellUpdate(self, x, y):
        
        cellState = self.grid[x][y]
        fundState = self.fundGrid[x][y]

        if cellState == 0: # if inanctive cell

            # activation of inactive cell
            if np.random.uniform() < self.pe: # random entering
                
                # decide cell type
                fundState = 1 if np.random.uniform() < (1-self.p_im) else 2                
                
                # decide cell state
                if fundState == 1:
                    cellState = 1 if (self.Price - self.fundPrice) <= 0 else -1
                else:
                    cellState = 1 if np.random.uniform() < 0.5 else -1
            else: # if not random entering
            
                neighbours = 0
                if abs(self.grid[x-1,y]) == 1:
                    neighbours += 1
                if abs(self.grid[x,y-1]) == 1:
                    neighbours += 1
                if abs(self.grid[(x+1)%self.width,y]) == 1:
                    neighbours += 1
                if abs(self.grid[x,(y+1)%self.height]) == 1:
                    neighbours += 1
        
                activate = (1-(1-self.ph)**neighbours) # activation probability
                if np.random.uniform() < activate: # activation by neighbour
                
                    # decide cell type
                    fundState = 1 if np.random.uniform() < (1-self.p_im) else 2                
                    
                    # decide cell state
                    if fundState == 1:
                        cellState = 1 if (self.Price - self.fundPrice) <= 0 else -1
                    else:
                        cellState = 1 if np.random.uniform() < 0.5 else -1
                        
        elif abs(cellState) == 1: # active cell

            if np.random.uniform() < self.pd: # random diffusion
                cellState = 0 # set inactive
                fundState = 0 # remove cell type
            else:
                k = self.cluster[x][y] # cluster number

                pk = self.localPRule(k, fundState)
                
                # pick spinstate
                cellState = 1 if np.random.uniform() < pk else -1
                    
        self.gridNext[x][y] = cellState
        self.fundGrid[x][y] = fundState
        
        
    
    def updateState(self):  # loop over all cells 
        
        for i in range(self.width):
            for j in range(self.height):
                
                self.cellUpdate(i, j)
    
    def run(self):
        
        for i in range(self.steps):
            
            # get cluster info
            self.cluster, self.clusterSize, self.nClust, self.clusterOnes = self.calcCluster()
            self.Xi_list = np.random.uniform(-1,1, self.nClust)
            
            self.Price = self.Price_List[-1] # get current price
           
            # update grid, save grid
            self.updateState()
            self.grid = self.gridNext.copy()
            
            if i % int(self.steps/self.nView) == 0:
                cmap = mpl.colors.ListedColormap(['red','white','green'])
                bounds=[-1.1,-.1,.1,1.1]
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
                plt.imshow(self.gridNext.astype(int),interpolation='nearest', cmap = cmap,norm=norm)
    
                plt.show()
            
            # update price
            self.x = self.updatePrice()
            
            # keep track of lists
            self.Price_List = np.append(self.Price_List, self.Price)
            self.logReturn = np.append(self.logReturn, self.Price_List[-1] - self.Price_List[-2])
            self.nlogReturn = np.append(self.nlogReturn, (self.logReturn[-1] - np.mean(self.logReturn))/np.std(self.logReturn))
        
            self.xChange = np.append(self.xChange, self.x)
            
            self.activeTraders = np.append(self.activeTraders, np.sum(np.absolute(self.gridNext))/(1.0*self.width*self.height)  )
            
            
        # calculate dynamics
        self.volatility = np.array([])
        for i in range(1,self.Price_List.size): # compute volatility
            ki = 400 if 400 < i else i
            price_avg = np.mean(self.Price_List[i-ki:i+1])
            value = np.sum(np.absolute(self.Price_List[i-ki:i+1] - price_avg))/(1.0*ki*price_avg)
            self.volatility = np.append(self.volatility, value)
         
        self.x_clustering = np.array([])
        self.v_clustering = np.array([])
        for lag in range(1,500): # array of correlation
            self.x_clustering= np.append(self.x_clustering,  np.sum(np.multiply(self.xChange[lag:],self.xChange[:-lag])))
            self.v_clustering = np.append(self.v_clustering,  np.sum(np.multiply(self.volatility[lag:],self.volatility[:-lag])))
        self.x_clustering = self.x_clustering/self.x_clustering[0] # normalize to first entry
        self.v_clustering = self.v_clustering/self.v_clustering[0] # normalize to first entry
        
        plt.figure()
        plt.plot(self.Price_List)
        plt.xlabel('time')
        plt.ylabel('Stock Price')
        plt.show()        
        
        plt.figure()
        plt.plot(self.activeTraders)
        plt.xlabel('time')
        plt.ylabel('fraction active traders')
        plt.show()
        
        plt.figure()
        plt.plot(self.nlogReturn)
        plt.xlabel('time')
        plt.ylabel('normalized log returns')
        plt.show()
        
        plt.figure()
        p1, = plt.plot(self.x_clustering, label='price clustering')
        p2, = plt.plot(self.v_clustering, label='volatility clustering')
        p3, = plt.plot(sp500_price_clustering, label='sp500 price clustering')
        p4, = plt.plot(sp500_volatility_clustering, label='sp500 volatility clustering')
        plt.legend(handles=[p1, p2, p3, p4], bbox_to_anchor=(1.05, 1), loc=2)
        plt.xlabel('Lag [-]')
        plt.ylabel('Correlation coeficient [-]')
        plt.show() 
        
        plt.figure()
        mu, sigma = np.mean(self.nlogReturn), np.std(self.nlogReturn)
        xmin, xmax = min(np.amin(self.nlogReturn),-7) , max(np.amax(self.nlogReturn), 7)
        x = np.linspace(xmin, xmax, 100)
        p1, = plt.plot(x,mlab.normpdf(x,0,1), label='normal distribution')
        
        hist, bin_edges = np.histogram(self.nlogReturn/sigma - mu, bins=20, normed=True)
        bin_means = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(hist))]
        p2 = plt.scatter(bin_means, hist, marker='o', label='model')
        
        hist, bin_edges = np.histogram(sp500_nlogReturns, bins=20, normed=True)
        bin_means = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(hist))]
        p3 = plt.scatter(bin_means, hist, marker='v', color='g', label='sp500')
        
        plt.legend(handles=[p1, p2, p3], bbox_to_anchor=(1.05, 1), loc=2)
        plt.yscale('log')
        plt.xlim([-7, 7])
        plt.xlabel('normalized logreturn values [-]')
        plt.ylabel('Probability [-]')
        plt.show()

# read in S&P 500 data set
path = "table.csv"
print "reading data..."
sp500 = pd.read_csv(path, sep=',')
print "done"           

# compute returns S&P 500, price clustering and volatility clustering
sp500_open = sp500[['Open']].as_matrix().flatten()
sp500_close = sp500[['Close']].as_matrix().flatten()

# normalized log returns for distribution analysis
sp500_logReturns = np.log(sp500_close) - np.log(sp500_open)
sp500_nlogReturns = np.array([])
for i in range(1, sp500_logReturns.size):
    val = (sp500_logReturns[i] - np.mean(sp500_logReturns[:i+1]))/np.std(sp500_logReturns[:i+1])
    sp500_nlogReturns = np.append(sp500_nlogReturns, val)
    
mu, sigma = np.mean(sp500_nlogReturns), np.std(sp500_nlogReturns)
sp500_nlogReturns = sp500_nlogReturns/sigma - mu

# price and volatility clustering
sp500_price_change = np.divide((sp500_close[:-1] - sp500_open[1:]),sp500_open[1:])
sp500_volatility = np.array([])
for i in range(1,sp500_open.size): # compute volatility
    ki = 400 if 400 < i else i
    price_avg = np.mean(sp500_open[i-ki:i+1])
    value = np.sum(np.absolute(sp500_open[i-ki:i+1] - price_avg))/(1.0*ki*price_avg)
    sp500_volatility = np.append(sp500_volatility, value)

sp500_price_clustering = np.array([])
sp500_volatility_clustering = np.array([])
for lag in range(1,500): # array of correlation with certain lags
    sp500_price_clustering = np.append(sp500_price_clustering,  np.sum(np.multiply(sp500_price_change[lag:],sp500_price_change[:-lag])))
    sp500_volatility_clustering = np.append(sp500_volatility_clustering,  np.sum(np.multiply(sp500_volatility[lag:],sp500_volatility[:-lag])))
sp500_price_clustering = sp500_price_clustering/sp500_price_clustering[0] # normalize to first entry
sp500_volatility_clustering = sp500_volatility_clustering/sp500_volatility_clustering[0] # normalize to first entry    
          

width = 50 # grid width
height = 100 # grid height
steps = 9000 # number of steps
nView = 5 # number of grid plots
initProb = 0.55 # probability of active cell in initialization
initPrice = 100.0 # initial price
fundPrice = 100.0 # fundamental price
p_im = 0.0 # probability of imitator type (2)
pe = 0.0001 # enter probability
pd = 0.05 # diffusion probability
ph = 0.0485/1.5 # neighbour activation probability

model = CAmodel(width, height, steps, nView, initProb, initPrice, fundPrice, p_im, pe, pd, ph)
model.run()