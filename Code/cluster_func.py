import numpy as np
from scipy.ndimage import measurements

# input trader activity field at time t
# returns: 
#       - lw: field with cluster labels, inactive traders are labeled as 0
#       - area: array with area size of each cluster, cluster label corresponds to area index
#       - num: total number of clusters
#       - cluster_ones: number of +1 in cluster

def cluster(field):
    
    field_abs = np.absolute(field) # reduce field to active/inactive traders
    field_abs = field_abs == 1 # get field of True/False values
    lw, num = measurements.label(field_abs) # lw: matrix with cluster numbers, num: total number of clusters 
    area = measurements.sum(field_abs, lw, index=np.arange(1,num+1)) # area: matrix of cluster size
    
    cluster_ones = np.zeros(num) # define empty array
    for i in range(1,num+1): # loop clusters
        cluster_ones[i-1] = (np.where(np.logical_and(lw==i,field==1))[0]).size # get numberof +1 in cluster
       
    return lw, area, num, cluster_ones
 
## test  script  
#f = np.random.choice([0,-1,1], 10*10, [0.8,0.1,0.1]).reshape(10,10)
#a,b,c,d = cluster(f)
#print f
#print a
#print b
#print c
#print d
  
  

    
