from numpy import zeros, array

#function that creates alpha according to the chosen regularization kernel
def create_alpha(RegularizationKernel,sig):
    l = 0.8
    p = 0.5
    c = 1   
    s = sig

    if(RegularizationKernel=='DC'):
        alpha_init = array([c,l,p,s])
        return alpha_init
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC' or RegularizationKernel=='SS'):
        alpha_init = array([c,l,s])     
        return alpha_init         
    elif(RegularizationKernel=='none'):
        return None

#function to create the bounds of the minimization according to the chosen regularization kernel
def create_bounds(RegularizationKernel):
    if(RegularizationKernel=='DC'):
        bnds = ((1e-8, None), (0.72, 0.99), (-0.99, 0.99), (0, None))
        return bnds
    elif(RegularizationKernel=='DI' or RegularizationKernel=='TC'):
        bnds = ((1e-8, None), (0.7, 0.99), (0, None))
        return bnds
    elif(RegularizationKernel=='SS'):
        bnds = ((1e-8, None), (0.9, 0.99), (0, None))
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
