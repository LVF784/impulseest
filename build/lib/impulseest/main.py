from numpy import zeros, identity, transpose, std, array, hstack, bmat, log, vstack, array, prod, mean, diag, sign, sqrt, outer
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

    u = u.reshape(len(u),1)
    y = y.reshape(len(y),1)
    N = len(y)
    
    argument_check(u,y,n,N,PreFilter,RegularizationKernel,MinimizationMethod)

    if(PreFilter!='none'):
        u = whiten(u,method=PreFilter)
        y = whiten(y,method=PreFilter)

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
                    None            
        return P

    #precomputation for cost function (algorithm2)
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
        cost = (r**2)/(sig**2) + (N-n)*log(sig**2) + 2*log(det(R1)+1e-6)
        return cost

    #minimize cost function to estimate the impulse response
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
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        alpha_init = array([c,l])     
        return alpha_init         
    elif(RegularizationKernel=='none'):
        return None

def create_bounds(RegularizationKernel):
    if(RegularizationKernel=='DC'):
        bnds = ((1e-8, None), (0.72, 0.99), (-0.99, 0.99))
        return bnds
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        bnds = ((1e-8, None), (0.7, 0.99))
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
        raise Exception("the chosen prewhitening method is not valid.")

    if(MinimizationMethod not in ['Powell', 'TNC', 'L-BFGS-B']):
        raise Exception("the chosen minimization method is not valid. Check scipy.minimize.optimize documentation for bound-constrained methods.")

    return None

def whitening_matrix(X, assume_centered=False, method='cholesky', fudge=1e-8):   
    # Make sure data is n_samples x n_features
    X = X.reshape((-1, prod(X.shape[1:])))

    # Center
    X_centered = X
    if not assume_centered:
        X_centered = X - mean(X, axis=0)

    cov = X_centered.T @ X_centered / X_centered.shape[0]

    if method in ['zca', 'pca', 'cholesky']:
        U, sigma, _ = svd(cov)
        U = U @ diag(sign(diag(U)))  # Fix sign ambiguity
        invsqrt_sigma = diag(1.0 / sqrt(sigma + fudge))
        if method == 'zca':
            W = U @ invsqrt_sigma @ U.T
        elif method == 'pca':
            W = invsqrt_sigma @ U.T
        elif method == 'cholesky':
            W = cholesky(U @ diag(1.0 / sigma) @ U.T,)
    elif method in ['zca_cor', 'pca_cor']:
        stds = sqrt(diag(cov))
        corr = cov / outer(stds, stds)
        G, theta, _ = svd(corr)
        G = G @ diag(sign(diag(G)))  # Fix sign ambiguity
        invsqrt_theta = diag(1.0 / sqrt(theta + fudge))
        if method == 'zca_cor':
            W = G @ invsqrt_theta @ G.T @ diag(1 / stds)
        elif method == 'pca_cor':
            W = invsqrt_theta @ G.T @ diag(1 / stds)
    else:
        raise ValueError(f'Whitening method {method} not found.')

    return W


def whiten(X, assume_centered=False, method='cholesky', fudge=1e-8):
    # Center
    X_centered = X
    if not assume_centered:
        X_centered = X - mean(X, axis=0)

    W = whitening_matrix(
        X_centered, assume_centered=True, method=method, fudge=fudge
    )
    Z = X_centered @ W.T
    Z = Z.reshape(X.shape)
    
    return Z