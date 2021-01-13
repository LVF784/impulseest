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
#plant1---------------------------------------------------------
#---------------------------------------------------------------
z1g = 0.8
p1g = 0.9 + 0.3j
p2g = np.conj(p1g)
numG1 = [1,-z1g]
denG1 = [1,float(-(p1g+p2g)),float(p1g*p2g)]
evalg = sum(numG1)/sum(denG1)
numG1 = np.dot(numG1,(1/evalg))
G1 = ss.TransferFunction(numG1,denG1,dt=Ta)

#---------------------------------------------------------------
#plant2---------------------------------------------------------
#---------------------------------------------------------------
z1g = 1.2
p1g = 0.96
numG2 = np.dot(-1,[1,-z1g])
denG2 = [1,-p1g]
evalg = sum(numG2)/sum(denG2)
numG2 = np.dot(numG2,(1/evalg))
G2 = ss.TransferFunction(numG2,denG2,dt=Ta)

#---------------------------------------------------------------
#plant3---------------------------------------------------------
#---------------------------------------------------------------
z1g = 1.2
numG3 = np.dot(-1,[1,-z1g])
denG3 = [1,-1.88,0.9032]
evalg = sum(numG3)/sum(denG3)
numG3 = np.dot(numG3,(1/evalg))
G3 = ss.TransferFunction(numG3,denG3,dt=Ta)

#---------------------------------------------------------------
#plant4---------------------------------------------------------
#---------------------------------------------------------------
z1g = 0.5
p1g = 0.3333
numG4 = np.dot(0.86193,[1,-z1g])
denG4 = [1,-0.3333]
evalg = sum(numG4)/sum(denG4)
numG4 = np.dot(numG4,(1/evalg))
G4 = ss.TransferFunction(numG4,denG4,dt=Ta)

#---------------------------------------------------------------
#input-output signals-------------------------------------------
#---------------------------------------------------------------
u = r
t,y1 = ss.dlsim(G1,u,t)
t,y2 = ss.dlsim(G2,u,t)
t,y3 = ss.dlsim(G3,u,t)
t,y4 = ss.dlsim(G4,u,t)

nu = 0.1*np.random.normal(0, .1, u.shape)
ny1 = 0.2*np.random.normal(0, .1, y1.shape)
ny2 = 0.2*np.random.normal(0, .1, y2.shape)
ny3 = 0.2*np.random.normal(0, .1, y3.shape)
ny4 = 0.2*np.random.normal(0, .1, y4.shape)

u = u + nu
y1 = y1 + ny1
y2 = y2 + ny2
y3 = y3 + ny3
y4 = y4 + ny4

y_list = [y1, y2, y3, y4]
G_list = [G1, G2, G3, G4]

#---------------------------------------------------------------
#run------------------------------------------------------------
#---------------------------------------------------------------

#regularized estimation test
print("\n")
print("REGULARIZED ESTIMATION TEST")
i = 0
for y in y_list:
    i = i+1
    print("\n")
    print("Plant {} results:" .format(i))

    G = G_list[i-1]
    t,G_h = ss.dimpulse(G, n=100)
    ir_real = np.squeeze(G_h)

    for reg in ['DC','DI','TC']:
        
        start_time = time.time()
        ir_est = impulseest(u,y,n=100,RegularizationKernel=reg)
        end_time = time.time()
        
        #calculating MSE
        sub = np.zeros(len(ir_est))
        for k in range(len(ir_est)):
            sub[k] = (ir_real[k]-ir_est[k])**2
        square = np.square(sub)
        summer = np.sum(square)
        mse = (1/len(ir_est))*summer        
        
        print("An MSE of {} was obtained using {} kernel and it took {:2f} seconds." .format(mse,reg,(end_time-start_time)))

#unregularized estimation test
print("\n")
print("UNREGULARIZED ESTIMATION TEST")
i = 0
for y in y_list:
    i = i+1
    print("\n")
    print("Plant {} results:" .format(i))

    G = G_list[i-1]
    t,G_h = ss.dimpulse(G, n=100)
    ir_real = np.squeeze(G_h)

    for pre in ['none','pca','pca_cor','zca','zca_cor','cholesky']:
        
        start_time = time.time()
        ir_est = impulseest(u,y,n=100,PreFilter=pre,RegularizationKernel='none')
        end_time = time.time()
        
        #calculating MSE
        sub = np.zeros(len(ir_est))
        for k in range(len(ir_est)):
            sub[k] = (ir_real[k]-ir_est[k])**2
        square = np.square(sub)
        summer = np.sum(square)
        mse = (1/len(ir_est))*summer        
        
        print("An MSE of {} was obtained using the {} prewhitening filter." .format(mse,pre))