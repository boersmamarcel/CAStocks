# ? %% no need to have stocks before selling? no
# ? %% you can sell/buy more than 1 at once? dunno
# ? %% difference between "transaction quantity" q^{t+1}_i,im/fu and V^{t+1}_i,im 
# done %% phi^t_fu/im is the same for all fundamentalists (i.e. gaussian random)
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.ndimage import measurements

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
        self.LogRet = np.zeros(steps)
        self.nLogRet = np.zeros(steps)
        self.steps = steps
        self.F = F
        self.width = width
        self.height = height
        self.actT = 1

        self.qpos = np.zeros(steps) # sum of positive values
        self.qneg = np.zeros(steps) # sum of negative values
        self.numberpos = np.zeros(steps) # number of positive q's
        self.numberneg = np.zeros(steps) # number of negative q's
        self.qtotal = np.zeros(steps) # sum of all q's
        # im

    # done    
    def getGrid(self):
        return self.grid

    def getActGrid(self):
        price = self.P[self.actT-1]
        interq = np.round(self.qgrid/price,decimals=0)
        # interq = self.qgrid
        return interq

    # done
    def getPrice(self):
        return self.P,self.LogRet,self.nLogRet
    
    # done
    def initializeGrid(self, pim, width, height):
        print "initialize grid"
        # grid: True if fundamentalist, False if imitator
        self.grid = (np.random.random(width*height).reshape(width,height)>pim)
        self.qgrid = np.zeros((width,height)) # grid of q-values
        self.qgridNew = np.zeros((width,height)) # next q-values
        self.Vgrid = np.zeros((width,height)) # grid of V-values
        self.VgridNew = np.zeros((width,height)) # nexct V-values
        return
    
    # done
    # the calculation of L
    def calcL(self):
        # print "calculate L"
        k = self.k
        kmax = self.actT
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
        k = np.sum(self.P!=0)
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
            V = (self.F*self.eta(fu)-self.P[self.actT-1])
            return V
        else:
            q = 0.
            for a in range(-1,2):
                for b in range (-1,2):
                    if (abs(a) | abs(b)):
                        q += self.Vgrid[(i+a)%self.width,(j+b)%self.height]/8.
            return q
    
        
    # done
    def doStep(self,i):
        global im, fig
        if i == self.steps-1:
            print 'jaaaaaaa'
            plt.close(fig)
        
        (width, height) = self.grid.shape
        time = self.calck()
        self.actT = time

        self.phi_fu = np.random.normal(0,1)
        self.phi_im = np.random.normal(0,1)

        for w in range(width): # calculate new grid
            for h in range(height):
                self.doCellStep(w,h)

        # save updated grids
        self.Vgrid = self.VgridNew
        self.qgrid = self.qgridNew

        # update price and save change information
        self.updatePrice()
        self.saveInfo(time)
        
        # return stuff for animation
        A = self.getActGrid()
        im.set_data(A)

        return im
        
    # done
    def doCellStep(self, i, j):
        fu = self.grid[i,j]
        self.VgridNew[i,j] = self.calcV(i,j,fu)
        if fu:
            self.qgridNew[i,j] = self.VgridNew[i,j]*self.calcM()
        else:
            self.qgridNew[i,j] = self.VgridNew[i,j]*self.eta(fu)*self.calcM()
        return
        
    def saveInfo(self,t):
        self.qpos[t-1] = np.sum(self.qgrid[self.qgrid>0.])
        self.qneg[t-1] = np.sum(self.qgrid[self.qgrid<0.])
        self.numberpos[t-1] = np.sum(self.qgrid>0.)
        self.numberneg[t-1] = np.sum(self.qgrid<0.)
        self.qtotal[t-1] = np.sum(self.qgrid)



    # done    
    def updatePrice(self):
        t = self.actT
        self.P[t] = self.P[t-1] + self.c_p*np.sum(self.qgrid)/(self.width*self.height)
        self.LogRet[t-1] = np.log(self.P[t]) - np.log(self.P[t-1])
        self.nLogRet[t-1] = (self.LogRet[t-1] - np.mean(self.LogRet))/np.std(self.LogRet)
        # print "price updated"
        return

    def getData(self):
        return self.qpos, self.qneg, self.numberpos, self.numberneg, self.qtotal

import matplotlib as mpl
import matplotlib.animation as animation
global fig
fig = plt.figure()
skip = 10
saveVideo = False

cmap = mpl.colors.ListedColormap(['red','white','green'])
bounds=[-2000,-.1,.1,2000]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# writer, needed for saving the video
# ffmpeg needed, can be downloaded from: http://ffmpegmac.net (for mac)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='MarcelBoersma&AlexDeGeus&RoanVanLeeuwen'), bitrate=1800)

# def init():
    # im.set_data(self.ActGrid())
# def animate(i):
#     A = self.getActGrid()
#     im.set_data(A)
#     return im

#%matplotlib inline

steps = 60
p_im = 0.8 # probability that imitator
initPrice = 100 # initial price
F = 100 # Fundamental price
c_im = 0.7 # imitators constant
c_fu = 0.2 # fundamentalists constant
c_p = 0.005 # constant for price updating sensitity
Nx = 20 # width
Ny = 20 # height
k = 400 # k value
c_l = 20 # c_l
L_m = 0.01
model = CAStochastic(Nx, Ny, p_im, steps, initPrice, F, c_im, c_fu, c_p, c_l, L_m, k)
global im
im = plt.imshow(model.getActGrid(),interpolation='nearest',
                    cmap = cmap,norm=norm)
def animit():
    global im
    im.set_data(np.zeros((Nx,Ny)))
i = 0
# for i in range(0,steps):
anim = animation.FuncAnimation(fig, model.doStep, frames=steps, interval=1,init_func=animit)
plt.show()
    # model.doStep()
    # print i

P,lP,nlP = model.getPrice()
print P
print "the minimumvalue is", round(min(P),4)
print "the maximumvalue is", round(max(P),4)
print "where the average is", round(np.mean(P),4)

# plt.figure()
# plt.plot(P)
# plt.xlabel('time in steps of 1 [-]')
# plt.ylabel('Price [-]')
# plt.show()
fig2 = plt.figure()
ax = fig2.add_subplot(121)
hist,bin = np.histogram(nlP)#,bins=10)
ax.plot(hist)
ax.set_xscale('log')
ax.set_yscale('log')
# plt.show()

fax = fig2.add_subplot(122)
fax.hist(nlP,normed=1)
fax.set_yscale('log')
fax.set_xscale('log')
plt.show()

q1,q2,q3,q4,qs = model.getData()

x = range(0,40)
if False:
    plt.figure()
    plt.plot(x,q1,'g-',x,q2,'r-')
    plt.figure()
    plt.plot(x,q3,'g-',x,q4,'r-')
    plt.figure()
    plt.plot(x,qs,'k-')
    plt.show()
    plt.figure()
    plt.plot(lP)
    plt.figure()
    plt.plot(nlP)
    plt.show()


if saveVideo:
    anim.save('activityAnimation.mp4', writer=writer)
