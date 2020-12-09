import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import time

from impulseest import impulseest
from scipy.optimize import minimize
from random import choice

#---------------------------------------------------------------
#signal---------------------------------------------------------
#---------------------------------------------------------------
N = 1000
Ta = 1*10**(-3)

#creating prbs signal
def prbs():
    while True:
        yield choice([False,True])

r = np.zeros(N)
i = 0
for value in prbs():
    r[i] = value
    i = i+1
    if i==N:
        break

r = 2*r-1

r = np.concatenate((np.zeros(10), r, np.zeros(10)))
t = np.linspace(0,(N+20-1)*Ta,N+20)

#---------------------------------------------------------------
#plant----------------------------------------------------------
#---------------------------------------------------------------
kg = 2
z1g = 0.8
p1g = 0.92 + 0.3j
p2g = np.conj(p1g)
numG = [1,-z1g]
denG = [1,float(-(p1g+p2g)),float(p1g*p2g)]
evalg = sum(numG)/sum(denG)
numG = np.dot(numG,(1/evalg))
G = ss.TransferFunction(numG,denG,dt=Ta)

u = r
t,y = ss.dlsim(G,u,t)

nu = 0.1*np.random.normal(0, .1, u.shape)
ny = 0.2*np.random.normal(0, .1, y.shape)

u = u + nu
y = y + ny

#---------------------------------------------------------------
#ir-------------------------------------------------------------
#---------------------------------------------------------------

ir = impulseest(u,y,n=100,RegularizationKernel='DC')
t,G_h = ss.dimpulse(G, n=100)

plt.figure(1)
plt.plot(np.squeeze(G_h),color='C0')
plt.plot(ir,linestyle='--',color='C1')
plt.legend(['Real IR','Estimated IR'])
plt.grid()
plt.show()