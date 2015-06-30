import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import matplotlib.animation as animation
import matplotlib.mlab as mlab
import pandas as pd

class CAStochastic(object):
    
    def __init__(self, width, height, p_im, c_im, c_fu, c_p, c_l, k, L_m, initPrice, F, steps):
        
        self.steps = steps # number of iterations        
        
        # grid dimensions
        self.width = width
        self.height = height
        self.N = width*height # number of traders
        
        # probability of cell being an imitator
        self.p_imfu = p_im
        
        # grid of fundamentalists and imitators
        self.grid_imfu = np.random.uniform(0,1,width*height).reshape(width,height) > p_im
        self.fu_idx = np.where(self.grid_imfu) # remember indeces for updating
        self.im_idx = np.where(np.logical_not(self.grid_imfu))
        
        # V en q grids
        self.V = np.zeros((width, height))
        self.Vnext = np.zeros((width, height))
        self.q = np.zeros((width, height))
        
        # constants
        self.c_im = c_im # eta_im = 1 + c_im*phi_im
        self.c_fu = c_fu # eta_fu = 1 + c_fu*phi_fu
              
        self.c_p = c_p # P_t+1 = P_t = c_p*Q_t/N
        self.c_l = c_l # M(c_l, L_t, L_m)
        
        self.k = k # Lt(k, P)
        self.L_m = L_m # M(c_l, L_t, L_m)
        
        self.initPrice = initPrice
        self.F = F
                
        # time dependend constants
        self.Q_t = 0 # P_t+1 = P_t = c_p*Q_t/N
        self.dp = 0
        
        self.L_t = 0 #  1/k sum (P - Pavg)/Pavg
        self.M_t = 0 # M(c_l, L_t, L_m)
        
        self.phi_fu = 0 # eta_im = 1 + c_im*phi_im
        self.phi_im = 0 # eta_fu = 1 + c_fu*phi_fu
        self.eta_fu = 0 # eta_im = 1 + c_im*phi_im
        self.eta_im = 0 # eta_fu = 1 + c_fu*phi_fu
        
        # first initialization control
        self.init = True
        
        # plotting lists
        self.Price_list = np.array([initPrice])
        self.logReturn = np.array([0])
        self.nlogReturn = np.array([0])
        self.Price_change = np.array([])
        self.Fluctation = np.array([])
        
     
    def calcEta(self): # set news sensitivity paramter
        
        self.phi_fu, self.phi_im = np.random.normal(0,1), np.random.normal(0,1)
        return 1 + self.c_fu*self.phi_fu, 1 + self.c_im*self.phi_im 
        
    def calcLt(self):
        
        time = self.Price_list.size
        kt = self.k if time < self.k else time
        Price_avg = np.mean(self.Price_list[time-kt:time+1])
        
        value = np.sum(np.absolute(self.Price_list[time-kt:time+1] - Price_avg))/(1.0*kt*Price_avg)
        return value 
       
    def calcMt(self):

        if self.L_t<=self.L_m:
            M = self.c_l*self.L_t
        else:
            M = self.c_l*(-self.L_t+2*self.L_m)
            
        return M if M >= 0.05 else 0.05
            
    def updatePrice(self):
        
        currentPrice = self.Price_list[-1]
        self.Q_t = np.sum(self.q)
        self.dp = self.c_p*self.Q_t/self.N
        nextPrice = currentPrice + self.dp
        return nextPrice if nextPrice >=0 else 0 # lower bound         
    
    def updateImitator(self, x, y)  :
        
        q = 0.
        for a in range(-1,2):
            for b in range (-1,2):
                if (abs(a) | abs(b)):
                    q += self.V[(x+a)%self.width,(y+b)%self.height]/8.
                    
        return q
    
    def doStep(self,i):
                
        self.eta_fu, self.eta_im = self.calcEta() # set eta parameters
        self.L_t = self.calcLt()
        self.M_t = self.calcMt()
        
        # update fundamentalists
        if self.init: # only for initialization
            self.V[self.fu_idx] =  self.F*self.eta_fu - self.Price_list[-1]
            self.init = False 
            
        self.Vnext[self.fu_idx] =  self.F*self.eta_fu - self.Price_list[-1]
        self.q[self.fu_idx] =  (self.F*self.eta_fu - self.Price_list[-1])*self.M_t
        
        # update imitators
        for i in range(self.im_idx[0].size):
            # location
            x = self.im_idx[0][i]
            y = self.im_idx[1][i]
            
            # get imitated V value
            Vnew = self.eta_im*self.updateImitator(x, y) 
            self.Vnext[x,y] = Vnew 
            
            self.q[x,y] = Vnew*self.M_t
            
        self.V = self.Vnext.copy() # save new grid
        
        nextPrice = self.updatePrice()
        self.Price_list = np.append(self.Price_list, nextPrice)
        self.logReturn = np.append(self.logReturn, self.Price_list[-1] - self.Price_list[-2])
        self.nlogReturn = np.append(self.nlogReturn, (self.logReturn[-1] - np.mean(self.logReturn))/np.std(self.logReturn))
        
        self.Price_change = np.append(self.Price_change, self.dp)
        self.Fluctation = np.append(self.Fluctation, self.L_t)
        
#        print self.eta_fu, self.eta_im, self.L_t, self.M_t, self.Price_list[-1]
        
        # only use when animating
#        fig.clf()
#        ax1 = fig.add_subplot(1,1,1)
#        cax = ax1.imshow(self.q, extent=[0, self.width, 0, self.height])
#        fig.colorbar(cax, ticks = [-2, 0, 2])
        
    def run(self):
        
        for i in range(self.steps):
            
            self.doStep(i)
            
        # make plots
        plt.figure()
        plt.plot(self.Price_list[1:])
        plt.show()

        plt.figure()
        mu, sigma = np.mean(self.nlogReturn), np.std(self.nlogReturn)
        xmin, xmax = min(np.amin(self.nlogReturn),-7) , max(np.amax(self.nlogReturn), 7)
        x = np.linspace(xmin, xmax, 100)
        p1, = plt.plot(x,mlab.normpdf(x,mu,sigma), label='normal distribution')
        
        hist, bin_edges = np.histogram(self.nlogReturn, bins=20, normed=True)
        bin_means = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(hist))]
        plt.scatter(bin_means, hist, marker='o', label='model')
        
        hist, bin_edges = np.histogram(sp500_nlogReturns, bins=20, normed=True)
        bin_means = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(hist))]
        plt.scatter(bin_means, hist, marker='+', label='sp500')
        
#        plt.legend(handles=[p1, p2, p3], bbox_to_anchor=(1.05, 1), loc=2)
        plt.yscale('log')
        plt.xlim([-6,6])
        plt.show()
        
        plt.clf()
        plt.imshow(self.q, extent=[0, self.width, 0, self.height])
        plt.colorbar()
        plt.show()
        
        plt.clf()
        plt.imshow(self.grid_imfu, extent=[0, self.width, 0, self.height])
#        plt.colorbar()
        plt.show()
        
        self.Price_correlation = np.array([])
        for lag in range(1,50): # array of correlation
            self.Price_correlation = np.append(self.Price_correlation,  np.sum(np.multiply(self.Price_change[lag:],self.Price_change[:-lag])))
        self.Price_correlation = self.Price_correlation/self.Price_correlation[0] # normalize to first entry
        
        self.v_clustering = np.array([])
        for lag in range(1,50): # array of correlation
            self.v_clustering = np.append(self.v_clustering,  np.sum(np.multiply(self.Fluctation[lag:],self.Fluctation[:-lag])))
        self.v_clustering = self.v_clustering/self.v_clustering[0] # normalize to first entry
        
        plt.figure()
        p1, = plt.plot(self.Price_correlation, label='price autocorrelation')
        p2, = plt.plot(self.v_clustering, label='volatility clustering')
        plt.legend(handles=[p1, p2], bbox_to_anchor=(1.05, 1), loc=2)
        plt.show()
        
# read in S&P 500 data set
path = "table.csv"
print "reading data..."
sp500 = pd.read_csv(path, sep=',')
print "done"

# compute returns S&P 500 
sp500_logReturns = (np.log(sp500[['Close']].as_matrix()) - np.log(sp500[['Open']].as_matrix())).flatten()
sp500_nlogReturns = np.array([])
for i in range(1, sp500_logReturns.size):
    val = (sp500_logReturns[i] - np.mean(sp500_logReturns[:i+1]))/np.std(sp500_logReturns[:i+1])
    sp500_nlogReturns = np.append(sp500_nlogReturns, val)

# model parameters     
width = 25 # width
height = 25 # height

p_im = 0.7 # probability that imitator

c_im = 0.7 # imitators constant
c_fu = 0.2 # fundamentalists constant

c_p = 0.005 # constant for price updating sensitity

c_l = 20 # c_l
k = 400 # k value
L_m = 0.01

initPrice = 100 # initial price    
F = 100 # Fundamental price  

steps = 1000

model = CAStochastic(width, height, p_im, c_im, c_fu, c_p, c_l, k, L_m, initPrice, F, steps) 
print "running model"    
model.run()   

# animate
#fig = plt.figure()
#
#ax1 = fig.add_subplot(1,1,1)
#ani = animation.FuncAnimation(fig, model.doStep, interval=1000)
#plt.show()
        
        
