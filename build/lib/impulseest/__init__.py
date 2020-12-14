from numpy import zeros, identity, transpose, std, array
from numpy.linalg import pinv, slogdet
from scipy.optimize import minimize

def impulseest(u, y, n=100, RegularizationKernel='none'):

    u = u.reshape(len(u),1)
    y = y.reshape(len(y),1)
    N = len(y)
    
    if(len(u)!=len(y)):
        raise Exception("u and y must have the same size.")       

    if(n>=N):
        raise Exception("n must be at least 1 sample smaller than the length of the signals.")

    if(RegularizationKernel=='DC'):
        None
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        None
    elif(RegularizationKernel=='CS'):
        None
    elif(RegularizationKernel=='none'):
        None
    else:
        raise Exception("the chosen regularization kernel is not valid.")

    Phi = create_Phi(u,n,N)
    Y = create_Y(y,n,N)
    
    #calculate impulse response without regularization
    ir_ls = pinv(Phi @ transpose(Phi)) @ Phi @ Y    

    #initialize variables for hyperparameter estimation
    I = identity(N-n)
    P = zeros((n,n))
    c = 0.5
    l = 0.5
    p = 0.5
    sig = std(ir_ls)/2
    

    if(RegularizationKernel=='DC'):
        alpha_init = array([c,l,p])
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        alpha_init = array([c,l])
    elif(RegularizationKernel=='CS'):
        alpha_init = array([c])

    bnds = create_bounds(RegularizationKernel)

    def Prior(alpha):   
        for k in range(n):
            for j in range(n):
                if(RegularizationKernel=='DC'):
                    P[k,j] = alpha[0]*(alpha[2]**abs(k-j))*(alpha[1]**((k+j)/2))
                elif(RegularizationKernel=='DI'):
                    if(k==j):
                        P[k,j] = alpha[0]*(alpha[1]**k)
                    else:
                        P[k,j] = 0
                elif(RegularizationKernel=='TC'):
                    P[k,j] = alpha[0]*min(alpha[1]**j,alpha[1]**k)
                elif(RegularizationKernel=='CS'):
                    if(k<=j):
                        P[k,j] = alpha[0]*((k**2)/2)*(j-(k/3))
                    else:
                        P[k,j] = alpha[0]*((j**2)/2)*(k-(j/3))                    
        return P

    def _slogdet(alpha):
        sign,logdet = slogdet(transpose(Phi) @ Prior(alpha) @ Phi + (sig**2)*I)
        return logdet

    def ML(alpha):
        return (transpose(Y) @ pinv((transpose(Phi) @ Prior(alpha) @ Phi + (sig**2)*I)) @ Y + _slogdet(alpha)).flatten()

    if(RegularizationKernel!='none'):
            A = minimize(ML, alpha_init, method='L-BFGS-B', options={'ftol': 1e-6}, bounds=bnds)
            alpha = A.x
            ir = Prior(alpha) @ Phi @ pinv((transpose(Phi) @ Prior(alpha) @ Phi + (sig**2)*I)) @ Y
    else:
        ir = ir_ls

    ir = ir.reshape(len(ir),1)
    return ir

def create_bounds(RegularizationKernel):
    if(RegularizationKernel=='DC'):
        bnds = ((0, None), (0, 1), (None,1))
        return bnds
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        bnds = ((0, None), (0, 1))
        return bnds
    elif(RegularizationKernel=='CS'):
        bnds = ((0,None),)
        return bnds
    elif(RegularizationKernel=='none'):
        return None

def create_Phi(u,n,N):
    Phi = zeros((n,N-n))
    for i in range(n):
        for j in range(N-n):
            Phi[i,j] = u[n+j-i]
    return Phi

def create_Y(y,n,N):
    Y = zeros((N-n,1))
    for i in range(N-n):
        Y[i,0] = y[n+i]
    return Y