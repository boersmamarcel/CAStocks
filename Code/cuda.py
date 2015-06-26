from timeit import default_timer as timer
import math
import numpy as np
import pylab
from numbapro import cuda, cudadrv
# For machine with multiple devices
cuda.select_device(0)

@cuda.jit(argtypes=['float32[:,:]','float32[:,:]'])
def test(mat, nn):
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	bx = cuda.blockIdx.x
	by = cuda.blockIdx.y
	bw = cuda.blockDim.x
	bh = cuda.blockDim.y

	x = tx + bx * bw
	y = ty + by * bh

	nn[x,y] = mat[x,y] +1



w = 100
h = 100
b = np.zeros((w,h),dtype=np.float32)
c = np.zeros((w,h),dtype=np.float32)

print b
test(b,c)
print c
