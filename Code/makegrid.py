# file: makegrid.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
fig = plt.figure()

# declare variables
N = 100 # size
skip = 10
saveVideo = False
A = np.zeros((N,N))
A = np.random.random_integers(-1,2,N**2).reshape(N,N) # array to be visualized

cmap = mpl.colors.ListedColormap(['red','white','green'])
bounds=[-1.1,-.1,.1,1.1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

im = plt.imshow(A,interpolation='nearest',
                    cmap = cmap,norm=norm)


# writer, needed for saving the video
# ffmpeg needed, can be downloaded from: http://ffmpegmac.net (for mac)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='MarcelBoersma&AlexDeGeus&RoanVanLeeuwen'), bitrate=1800)
im = plt.imshow(field)


def init():
	im.set_data(A)
def animate(i):
	global A
	A = np.random.random_integers(-1,2,N**2).reshape(N,N)
	im.set_data(A)
	return im

anim = animation.FuncAnimation(fig, animate, frames=4813/skip, interval=4)
if saveVideo:
	anim.save('gridAnimation.mp4', writer=writer)
plt.show()