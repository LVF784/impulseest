from numpy import zeros, identity, transpose, std, array, hstack, bmat, log, vstack, array, prod, mean, diag, sign, sqrt, outer, cov
from numpy.linalg import pinv, slogdet, cholesky, qr, det, svd 
from scipy.optimize import minimize

def impulseest(u, y, n=100, RegularizationKernel='none', PreFilter='none', MinimizationMethod='L-BFGS-B'):
    """Nonparametric impulse response estimation function

    This function estimates the impulse response with (optional) regularization.
    The variance increases linearly with the finite impulse response model order, 
    so this need to be counteracted by regularization. Prewhitening filtering is
    also optional. The six arguments in this function are:
    - u [numpy array]: input signal (size Nx1);
    - y [numpy array]: output signal (size Nx1);
    - n [int]: number of impulse response estimates (default is n=100);
    - RegularizationKernel [str]: regularization method ('DC','DI','TC','SS', default is 'none');
    - PreFilter [str]: prewhitening filter method ('zca', 'pca', 'cholesky', 'pca-cor', 'zca-cor', default is 'none');
    - MinimizationMethod: bound-constrained optimization method use to minimize the cost function ('Powell','TNC', default is 'L-BFGS-B').
   """

    #make sure u and y are shaped correctly
    u = u.reshape(len(u),1)
    y = y.reshape(len(y),1)
    N = len(y)  #length of input-output vectors
    
    #check the arguments entered by the user, raise exceptions if something is wrong
    argument_check(u,y,n,N,PreFilter,RegularizationKernel,MinimizationMethod)

    #if PreFilter is selected, then apply prewhitening filtering to u and y
    if(PreFilter!='none'):
        u = whiten(u,method=PreFilter)
        y = whiten(y,method=PreFilter)

    #arrange the regressors to least-squares according to T. Chen et al (2012)
    Phi = create_Phi(u,n,N)
    Y = create_Y(y,n,N)
    
    #calculate impulse response without regularization
    ir_ls = pinv(Phi @ transpose(Phi)) @ Phi @ Y    

    #initialize variables for hyper-parameter estimation
    I = identity(n)         #identitity matrix
    sig = std(ir_ls)        #sigma = standard deviation of the LS solution
    P = zeros((n,n))        #zero matrix

    #initialize alpha and choose bounds according to the chosen regularization kernel
    alpha_init = create_alpha(RegularizationKernel)
    bnds = create_bounds(RegularizationKernel)

    #function to create the regularization matrix
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
                    None            
        return P

    #precomputation for the Algorithm 2 according to T. Chen, L. Ljung (2013)
    aux0 = qr(hstack((transpose(Phi),Y)),mode='r')
    Rd1 = aux0[0:n+1,0:n]
    Rd2 = aux0[0:n+1,n]
    Rd2 = Rd2.reshape(len(Rd2),1)

    #cost function written as the Algorithm 2 presented in T. Chen, L. Ljung (2013)
    def algorithm2(alpha):
        L = cholesky(Prior(alpha))
        Rd1L = Rd1 @ L
        to_qr = bmat([[Rd1L,Rd2],[sig*I,zeros((n,1))]])
        R = qr(to_qr,mode='r')
        R1 = R[0:n,0:n]
        r = R[n,n]
        cost = (r**2)/(sig**2) + (N-n)*log(sig**2) + 2*log(det(R1)+1e-6)
        return cost

    #minimize Algorithm 2 to estimate the impulse response with scipy optimization
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

#function that creates alpha according to the chosen regularization kernel
def create_alpha(RegularizationKernel):
    l = 0.8
    p = 0.5
    c = 1     
    if(RegularizationKernel=='DC'):
        alpha_init = array([c,l,p])
        return alpha_init
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        alpha_init = array([c,l])     
        return alpha_init         
    elif(RegularizationKernel=='none'):
        return None

#function to create the bounds of the minimization according to the chosen regularization kernel
def create_bounds(RegularizationKernel):
    if(RegularizationKernel=='DC'):
        bnds = ((1e-8, None), (0.72, 0.99), (-0.99, 0.99))
        return bnds
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        bnds = ((1e-8, None), (0.7, 0.99))
        return bnds
    elif(RegularizationKernel=='none'):
        return None

#function to create the Phi regressor matrix
def create_Phi(u,n,N):
    Phi = zeros((n,N-n))
    for i in range(n):
        for j in range(N-n):
            Phi[i,j] = u[n+j-i]
    return Phi

#functino to create the Y regressor vector
def create_Y(y,n,N):
    Y = zeros((N-n,1))
    for i in range(N-n):
        Y[i,0] = y[n+i]
    return Y

#function to check all the arguments entered by the user, raising execption if something is wrong
def argument_check(u,y,n,N,PreFilter,RegularizationKernel,MinimizationMethod):
    if(PreFilter!='none' and RegularizationKernel!='none'):
        raise Exception("Prewhitening filter can only be used in the non-regularized estimation.")
    
    if(len(u)!=len(y)):
        raise Exception("u and y must have the same size.")

    if(n>=N):
        raise Exception("n must be at least 1 sample smaller than the length of the signals.")

    if(RegularizationKernel not in ['DC','DI','TC','none']):
        raise Exception("the chosen regularization kernel is not valid.")

    if(PreFilter not in ['zca', 'pca', 'cholesky', 'pca_cor', 'zca_cor', 'none']):
        raise Exception("the chosen prewhitening_matrixhitening method is not valid.")

    if(MinimizationMethod not in ['Powell', 'TNC', 'L-BFGS-B']):
        raise Exception("the chosen minimization method is not valid. Check scipy.minimize.optimize documentation for bound-constrained methods.")

    return None

def whiten(x, method='cholesky'):
    """Prewhitening of a discrete-time signal

    This function applies what is proposed in A. Kessy et al (2015)
    to whiten a discrete-time signal. The inputs are:
    - x [numpy array]: discrete-time signal (size Nx1);
    - method [str]: method used to create the W matrix, available options
    are: 'zca', 'pca', 'cholesky', 'zca_cor', 'pca_cor'.
    """
    x = x.reshape((-1, prod(x.shape[1:])))
    x = x - mean(x)
    W = create_whiteningmatrix(x, method=method)
    z = x @ W.T
    z = z.reshape(x.shape)
    
    return z

#function to create the whitening matrix according to A. Kessy et al (2015)
def create_whiteningmatrix(x, method='cholesky'):   
    covx = x.T @ x / len(x)

    if method in ['zca', 'pca', 'cholesky']:
        U, sigma, _ = svd(covx)
        U = U @ diag(sign(diag(U)))
        invsqrt_sigma = diag(1.0 / sqrt(sigma + 1e-8))
        if method == 'zca':
            W = U @ invsqrt_sigma @ U.T
        elif method == 'pca':
            W = invsqrt_sigma @ U.T
        elif method == 'cholesky':
            W = cholesky(U @ diag(1.0 / sigma) @ U.T,)
    elif method in ['zca_cor', 'pca_cor']:
        stds = sqrt(diag(covx))
        corr = covx / outer(stds, stds)
        G, theta, _ = svd(corr)
        G = G @ diag(sign(diag(G)))
        invsqrt_theta = diag(1.0 / sqrt(theta + 1e-8))
        if method == 'zca_cor':
            W = G @ invsqrt_theta @ G.T @ diag(1 / stds)
        elif method == 'pca_cor':
            W = invsqrt_theta @ G.T @ diag(1 / stds)
    else:
        raise Exception("Unvalid whitening method.")

    return W