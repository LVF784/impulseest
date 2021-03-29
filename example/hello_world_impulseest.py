import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import time
from impulseest import impulseest
from random import choice

#---------------------------------------------------------------
#signal---------------------------------------------------------
#---------------------------------------------------------------
N = 2000
Ts = 0.001

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
t = np.linspace(0,(N+20-1)*Ts,N+20)

#---------------------------------------------------------------
#plant----------------------------------------------------------
#---------------------------------------------------------------
G = ss.ZerosPolesGain([1.4812],[0.9726197+0.10215693j, 0.9726197-0.10215693j],1,dt=Ts)
G = G.to_tf()

#---------------------------------------------------------------
#input-output signals-------------------------------------------
#---------------------------------------------------------------
u = r
_,y = ss.dlsim(G,u,t)

#noise
nu = np.random.normal(0, .1, u.shape)
ny = (abs(max(y))*0.5)*np.random.normal(0, .1, y.shape)
u = u + nu
y = y + ny

#---------------------------------------------------------------
#impulse response-----------------------------------------------
#---------------------------------------------------------------
t,G_h = ss.dimpulse(G,n=300)
ir_real = np.squeeze(G_h)
ir_none = impulseest(u,y,n=300,RegularizationKernel='none')
reg = 'DC'

start_time = time.time()
ir_est = impulseest(u,y,n=300,RegularizationKernel=reg)
end_time = time.time()

#regularized estimation MSE
sub = np.zeros(len(ir_est))
for k in range(len(ir_est)):
    sub[k] = (ir_real[k]-ir_est[k])**2
square = np.square(sub)
summer = np.sum(square)
mse_est = (1/len(ir_est))*summer        

#non regularized estimation MSE
for k in range(len(ir_none)):
    sub[k] = (ir_real[k]-ir_none[k])**2
square = np.square(sub)
summer = np.sum(square)
mse_none = (1/len(ir_none))*summer    

print("\n")
print(f'An MSE of {mse_est} was obtained using {reg} kernel and it took {(end_time-start_time):.2f} seconds.')
print(f'An MSE of {mse_none} was obtained using no regularization kernel.')
print("\n")

#plotting and saving impulse responses
fig,ax1 = plt.subplots()
lns3 = ax1.plot(ir_none,label="Non regularized estimated IR",linewidth=1.5,color='C2')
lns2 = ax1.plot(ir_est,label="Regularized estimated IR",linestyle='dashed',linewidth=1.5,color='C1')
lns1 = ax1.plot(ir_real,label="Model-based IR",linestyle='dotted',linewidth=1.5,color='C0')
ax1.set_xlabel("Samples [p.u.]",fontsize=11)
ax1.set_ylabel("Impulse response value [p.u.]",fontsize=11)

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc='lower right')
ax1.grid()

plt.show()
#fig.savefig("ir_example.pdf", bbox_inches='tight') #uncomment line to save pdf with graphical results
