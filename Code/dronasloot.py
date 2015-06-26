# ? %% no need to have stocks before selling?
# ? %% you can sell/buy more than 1 at once?
# ? %% difference between "transaction quantity" q^{t+1}_i,im/fu and V^{t+1}_i,im
# done %% phi^t_fu/im is the same for all fundamentalists (i.e. gaussian random)
import numpy as np
import matplotlib.pyplot as plt

# from scipy.ndimage import measurements

class CAStochastic(object):
    
    def __init__(self, width, height, pim, steps, initPrice, F, cim, cfu, c_p, c_l, L_m, k):
        print "Initialize model"
        self.initializeGrid(pim, width, height)
        print self.grid
        self.c_im = cim # constant for the imitators
        self.c_fu = cfu # constant for the fundamentalist
        self.c_p = c_p
        self.c_l = c_l
        self.L_m = L_m

        # some parameters
        self.k = k
        self.P = np.zeros(steps+1)
        self.P[0] = initPrice
        self.F = F
        self.width = width
        self.height = height

    # done    
    def getGrid(self):
        return self.grid

    # done
    def getPrice(self):
        return self.P
    
    # done
    def initializeGrid(self, pim, width, height):
        print "initialize grid"
        # self.grid = np.random.binomial(1, p, width*height).reshape(width, height)
        # True if fundamentalist
        self.grid = (np.random.random(width*height).reshape(width,height)>pim)
        self.qgrid = np.zeros((width,height))
        self.qgridNew = np.zeros((width,height))
        self.Vgrid = np.zeros((width,height))
        self.VgridNew = np.zeros((width,height))
        return
    
    # done
    # the calculation of L
    def calcL(self):
        # print "calculate L"
        k = self.k
        kmax = self.calck()
        if k>kmax:
            k=kmax
        p_avg = self.PAvg(k,kmax)
        L = 0.
        for i in range(kmax-k,kmax+1):
            L += (1./k)*abs(self.P[i]-p_avg)/p_avg
        return L
    
    # done
    # calculates the amount of prices, i.e. the amount of price updates (minus 1) done
    def calck(self):
        k = sum(self.P>0.)
        return k


    # done
    # calculates the average price during last k price updates.
    def PAvg(self,k,kmax):
        # print "calculate the average of P"
        return np.mean(self.P[kmax-k:kmax])

    # done
    # calculates the M-term
    def calcM(self):
        # print "calculate M"
        Lm = self.L_m
        cL = self.c_l
        L = self.calcL()
        if L<=Lm:
            M = cL*L
        else:
            M = cL*(-L+2*Lm)
        return M
    
    # done
    # calculates the eta term, which adds the news factor
    def eta(self,fu):
        # print "calculate L"
        if fu:
            return (1 + self.c_fu*self.phi_fu)
        else:
            return (1 + self.c_im*self.phi_im)

    # broken, dont use (to save comp time)
    def calcq(self,fu):
        return self.calcV(fu)*self.calcM()

    # done

    def calcV(self,i,j,fu):
        # print "calculate V"
        if fu:
            V = (self.F*self.eta(fu)-self.P[self.calck()-1])
            return V
        else:
            q = 0.
            for a in range(-1,2):
                for b in range (-1,2):
                    if (abs(a) | abs(b)):
                        q += self.Vgrid[(i+a)%self.width,(j+b)%self.height]/8.
            return q
    
        
    # done
    def doStep(self):
        (width, height) = self.grid.shape
        
        self.phi_fu = np.random.normal(0,1)
        self.phi_im = np.random.normal(0,1)

        for w in range(width): # loop grid
            for h in range(height):
                self.doCellStep(w,h)

        self.Vgrid = self.VgridNew
        self.qgrid = self.qgridNew
        self.updatePrice()
        return
        
    # done
    def doCellStep(self, i, j):
        fu = self.grid[i,j]
        self.VgridNew[i,j] = self.calcV(i,j,fu)
        if fu:
            self.qgridNew[i,j] = self.VgridNew[i,j]*self.calcM()
        else:
            self.qgridNew[i,j] = self.VgridNew[i,j]*self.eta(fu)*self.calcM()

        # cell = self.grid[i, j]
        # (w,h) = self.grid.shape
               
        # if cell == 0:
        #     enter = np.random.binomial(1,self.pe,1)
            
        #     #random enter probability
        #     if enter == 1:
        #         self.nextGrid[i,j] = 1 
        #     else:
        #         #activate by neighbors
        #         neighbours = 0
        #         if i != 0 and abs(self.grid[i-1,j]) == 1:
        #             neighbours += 1
        #         if j != 0 and abs(self.grid[i,j-1]) == 1:
        #             neighbours += 1
        #         if i != w-1 and abs(self.grid[i+1,j]) == 1:
        #             neighbours += 1
        #         if j != h-1 and abs(self.grid[i,j+1]) == 1:
        #             neighbours += 1
                
        #         activated = np.random.binomial(1, (1-(1-self.ph)**neighbours))
        #         self.nextGrid[i,j] = activated
            
        # elif abs(cell) == 1:
        #     #diffuse?
        #     diffuse = np.random.binomial(1,self.pd,1)
            
        #     if diffuse == 1 :
        #         self.nextGrid[i,j] = 0
        #     else:
        #         #update status cell
        #         k = self.cluster[i,j] #which cluster?
        #         pk = self.localPRule(k) #cell pk
        #         state = np.random.choice([1,-1], 1, p=[pk,(1-pk)])
        #         self.nextGrid[i,j] = state
        return
        
    # done    
    def updatePrice(self):
        t = self.calck()
        self.P[t] = self.P[t-1] + self.c_p*sum(sum(self.qgrid))/self.width/self.height
        print "price updated"
        return

# import matplotlib as mpl
# import matplotlib.animation as animation
# fig = plt.figure()

#%matplotlib inline

steps = 40
p_im = 0.5 # probability that imitator
initPrice = 100 # initial price
F = 100 # Fundamental price
c_im = 0.7 # imitators constant
c_fu = 0.2 # fundamentalists constant
c_p = 0.005 # constant for price updating sensitity
Nx = 512 # width
Ny = 128 # height
k = 400 # k value
c_l = 20 # c_l
L_m = 0.01
model = CAStochastic(Nx, Ny, p_im, steps, initPrice, F, c_im, c_fu, c_p, c_l, L_m, k)

i = 0
for i in range(0,steps):
    model.doStep()
    print i
print model.getPrice()


plt.plot(model.getPrice())
plt.xlabel('time in steps of 1 [-]')
plt.ylabel('Price [-]')
plt.show()

