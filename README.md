# impulseest() is a nonparametric impulse response estimation function relying solely on input-output data

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this situation by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] using the Empirical Bayes method (Carlin and Louis [1996]).

The six arguments in this function are: <br />
    - u [NumPy array]: input signal (size N x 1); <br />
    - y [NumPy array]: output signal (size N x 1); <br />
    - n [integer]: number of impulse response estimates (default is n = 100); <br />
    - RegularizationKernel [string]: regularization method - 'none', 'DC','DI','TC' (default is 'none'); <br />
    - MinimizationMethod [string]: bound-constrained optimization method used to minimize the cost function - 'L-BFGS-B', 'Powell','TNC' (default is 'L-BFGS-B').

The impulseest function returns a NumPy array of size n x 1 containing all the n impulse response estimates.

## Importing

```Python
from impulseest import impulseest
```

### Example

```Python
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import time

from impulseest import impulseest
from random import choice

#---------------------------------------------------------------
#signal---------------------------------------------------------
#---------------------------------------------------------------
N = 1000
Ta = 1*10**(-3)

#prbs
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
z1g = 0.8
p1g = 0.9 + 0.3j
p2g = np.conj(p1g)
numG1 = [1,-z1g]
denG1 = [1,float(-(p1g+p2g)),float(p1g*p2g)]
evalg = sum(numG1)/sum(denG1)
numG1 = np.dot(numG1,(1/evalg))
G = ss.TransferFunction(numG1,denG1,dt=Ta)

#---------------------------------------------------------------
#input-output signals-------------------------------------------
#---------------------------------------------------------------
u = r
t,y = ss.dlsim(G,u,t)

#noise
nu = 0.1*np.random.normal(0, .1, u.shape)
ny = 0.2*np.random.normal(0, .1, y.shape)

u = u + nu
y = y + ny

#---------------------------------------------------------------
#impulse response-----------------------------------------------
#---------------------------------------------------------------
t,G_h = ss.dimpulse(G, n=100)
ir_real = np.squeeze(G_h)

reg = 'DC'

start_time = time.time()
ir_est = impulseest(u,y,n=100,RegularizationKernel=reg)
end_time = time.time()

#MSE
sub = np.zeros(len(ir_est))
for k in range(len(ir_est)):
    sub[k] = (ir_real[k]-ir_est[k])**2
square = np.square(sub)
summer = np.sum(square)
mse = (1/len(ir_est))*summer        

print("\n")
print("An MSE of {} was obtained using {} kernel and it took {:.2f} seconds." .format(mse,reg,(end_time-start_time)))
print("\n")

#plotting result
plt.figure()
plt.plot(ir_real,color='C0')
plt.plot(ir_est,linestyle='--',color='C1')
plt.legend(['Real IR','Estimated IR'])
plt.grid()
plt.show()
```

## Contributor

Luan Vinícius Fiorio - vfluan@gmail.com
