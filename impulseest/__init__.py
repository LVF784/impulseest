from numpy import zeros, identity, transpose, std, array, hstack, bmat, log, vstack, array
from numpy.linalg import pinv, slogdet, cholesky, qr, det
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
    elif(RegularizationKernel=='SS'):
        None
    elif(RegularizationKernel=='none'):
        None
    else:
        raise Exception("the chosen regularization kernel is not valid.")

    Phi = create_Phi(u,n,N)
    Y = create_Y(y,n,N)
    
    #calculate impulse response without regularization
    ir_ls = pinv(Phi @ transpose(Phi)) @ Phi @ Y    

    #initialize variables for hyper-parameter estimation
    I = identity(n)
    sig = std(ir_ls)
    P = zeros((n,n))  

    alpha_init = create_alpha(RegularizationKernel)
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
                elif(RegularizationKernel=='SS'):
                    if(k>=j):
                        P[k,j] = alpha[0]*((alpha[1]**(2*k))/2)*(alpha[1]**j - ((alpha[1]**(k))/3))
                    else:
                        P[k,j] = alpha[0]*((alpha[1]**(2*j))/2)*(alpha[1]**k - ((alpha[1]**(j))/3))
                    None            
        return P

    #precomputation for alg2
    aux0 = qr(hstack((transpose(Phi),Y)),mode='r')
    Rd1 = aux0[0:n+1,0:n]
    Rd2 = aux0[0:n+1,n]
    Rd2 = Rd2.reshape(len(Rd2),1)

    def algorithm2(alpha):
        L = cholesky(Prior(alpha))
        Rd1L = Rd1 @ L
        to_qr = bmat([[Rd1L,Rd2],[sig*I,zeros((n,1))]])
        R = qr(to_qr,mode='r')
        R1 = R[0:n,0:n]
        r = R[n,n]
        cost = (r**2)/(sig**2) + (N-n)*log(sig**2) + 2*log(det(R1))
        return cost

    if(RegularizationKernel!='none'):
        A = minimize(algorithm2, alpha_init, method='L-BFGS-B', bounds=bnds)
        alpha = A.x
        L = cholesky(Prior(alpha))
        Rd1L = Rd1 @ L
        to_qr = bmat([[Rd1L,Rd2],[sig*I,zeros((n,1))]])
        R = qr(to_qr,mode='r')
        R1 = R[0:n,0:n]
        R2 = R[0:n,n]
        ir = L @ pinv(R1) @ R2
    else:
        ir = ir_ls

    ir = ir.reshape(len(ir),1)
    return ir

def create_alpha(RegularizationKernel):
    l = 0.8
    p = 0.5
    c = 1     
    if(RegularizationKernel=='DC'):
        alpha_init = array([c,l,p])
        return alpha_init
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC' or RegularizationKernel=='SS'):
        alpha_init = array([c,l])     
        return alpha_init         
    elif(RegularizationKernel=='none'):
        return None

def create_bounds(RegularizationKernel):
    if(RegularizationKernel=='DC'):
        bnds = ((1e-12, None), (0.72, 0.99), (-0.99, 0.99))
        return bnds
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        bnds = ((1e-12, None), (0.7, 0.99))
        return bnds
    elif(RegularizationKernel=='SS'):
        bnds = ((1e-12, None), (0.9, 0.99))
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