import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


data = pd.read_csv('table.csv', sep=',')
returns = data['Close']-data['Open']

values = []
for i in range(len(returns)):
	val = (returns[i] - np.mean(returns[1:i])/np.std(returns[1:i]))

	if np.isnan(val) == False and np.isinf(val) == False:
		values.append(val)

df = pd.DataFrame(values)

plt.figure()
plt.plot(values)
plt.show()

loc, scale = norm.fit(values)

count, bins, ignored = plt.hist(values, 50, normed=True)

print len(count)
print len(bins)

xs = []
for i in np.arange(1,len(bins)):
	xs.append((bins[i-1]+bins[i])/2)

plt.figure()
plt.plot(np.arange(-40, 40), norm.pdf(np.arange(-40,40), loc=loc, scale=scale))
plt.scatter(xs, count)
plt.yscale('log')
plt.show()

xcorrelation = []
for lag in range(1,50): # array of correlation
    xcorrelation = np.append(xcorrelation,  np.sum(np.multiply(returns[lag:],returns[:-lag])))
xcorrelation = xcorrelation/xcorrelation[0] # normalize to first entry

print "x-cor"
plt.figure()
plt.plot(xcorrelation)
plt.show()

# count, bins, ignored = plt.hist(values, 30, normed=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                 np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#           linewidth=2, color='r')
# plt.show()