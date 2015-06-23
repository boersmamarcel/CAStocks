import numpy as np
from scipy.ndimage import measurements

# input trader activity field at time t
# returns: 
#       - lw: field with cluster labels, inactive traders are labeled as 0
#       - area: array with area size of each cluster, cluster label corresponds to area index
#       - num: total number of clusters

def cluster(field):
    
    field = np.absolute(field) # reduce field to active/inactive traders
    field = field == 1 # get field of True/False values
    lw, num = measurements.label(field) # lw: matrix with cluster numbers, num: total number of clusters 
    area = measurements.sum(field, lw, index=np.arange(num + 1)) # get array of cluster size
    
    return lw, area, num
    

    
