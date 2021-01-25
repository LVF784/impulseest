from numpy import zeros, identity, transpose, std, hstack, bmat, log, convolve, diag
from numpy.linalg import pinv, qr, det, cholesky
from scipy.optimize import minimize

from impulseest.creation import create_alpha, create_bounds, create_Phi, create_Y

def impulseest(u, y, n=100, RegularizationKernel='none', BaseLine=False, MinimizationMethod='L-BFGS-B'):
    """Nonparametric impulse response estimation function

    This function estimates the impulse response with (optional) regularization.
    The variance increases linearly with the finite impulse response model order, 
    so this need to be counteracted by regularization. Prewhitening filtering is
    also optional. The six arguments in this function are:
    - u [numpy array]: input signal (size Nx1);
    - y [numpy array]: output signal (size Nx1);
    - n [int]: number of impulse response estimates (default is n=100);
    - RegularizationKernel [str]: regularization method ('DC','DI','TC','SS', default is 'none');
    - BaseLine [bool]: if True, a base-line model is used to identify most of the impulse response. Recommended to use in cases where the impulse response decays slowly;
    - MinimizationMethod[str]: bound-constrained optimization method use to minimize the cost function ('Powell','TNC', default is 'L-BFGS-B').
   """

    #make sure u and y are shaped correctly
    u = u.reshape(len(u),1)
    y = y.reshape(len(y),1)
    N = len(y)  #length of input-output vectors
    
    #check the arguments entered by the user, raise exceptions if something is wrong
    argument_check(u,y,n,N,RegularizationKernel,MinimizationMethod)

    #arrange the regressors to least-squares according to T. Chen et al (2012)
    Phi = create_Phi(u,n,N)
    Y = create_Y(y,n,N)
    
    #calculate impulse response without regularization
    ir_ls = pinv(Phi @ transpose(Phi)) @ Phi @ Y 
    ir_ls = ir_ls.reshape(len(ir_ls),1)   

    #initialize variables for hyper-parameter estimation
    I = identity(n)         #identitity matrix
    sig = std(ir_ls)        #sigma = standard deviation of the LS solution
    P = zeros((n,n))        #zero matrix

    #initialize alpha and choose bounds according to the chosen regularization kernel
    alpha_init = create_alpha(RegularizationKernel,sig)
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
                elif(RegularizationKernel=='SS'):
                    if(k>=j):
                        P[k,j] = alpha[0]*((alpha[1]**(2*k))/2)*((alpha[1]**j)-(alpha[1]**k)/3)
                    else:
                        P[k,j] = alpha[0]*((alpha[1]**(2*j))/2)*((alpha[1]**k)-(alpha[1]**j)/3)
                else:
                    None            
        return P

    if(RegularizationKernel!='none' and BaseLine==True):
        Yb = Phi.T @ ir_ls        
        Yr = Y - Yb

        #precomputation for the Algorithm 2 according to T. Chen, L. Ljung (2013)
        aux0 = qr(hstack((transpose(Phi),Yr)),mode='r')
        Rd1 = aux0[0:n+1,0:n]
        Rd2 = aux0[0:n+1,n]
        Rd2 = Rd2.reshape(len(Rd2),1)

        #cost function written as the Algorithm 2 presented in T. Chen, L. Ljung (2013)
        def algorithm2(alpha):
            L = cholesky(Prior(alpha))
            Rd1L = Rd1 @ L
            to_qr = bmat([[Rd1L,Rd2],[alpha[len(alpha)-1]*I,zeros((n,1))]])
            R = qr(to_qr,mode='r')
            R1 = R[0:n,0:n]
            r = R[n,n]
            cost = (r**2)/(alpha[len(alpha)-1]**2) + (N-n)*log(alpha[len(alpha)-1]**2) + 2*sum(log(abs(diag(R1))))    
            return cost

        A = minimize(algorithm2, alpha_init, method='L-BFGS-B', bounds=bnds)
        alpha = A.x
        L = cholesky(Prior(alpha))
        Rd1L = Rd1 @ L
        to_qr = bmat([[Rd1L,Rd2],[alpha[len(alpha)-1]*I,zeros((n,1))]])
        R = qr(to_qr,mode='r')
        R1 = R[0:n,0:n]
        R2 = R[0:n,n]
        ir_r = L @ pinv(R1) @ R2

        ir_r = ir_r.reshape(len(ir_r),1)
        ir_ls = ir_ls.reshape(len(ir_ls),1)

        ir = ir_ls + ir_r

    elif(RegularizationKernel!='none' and BaseLine==False):
        #precomputation for the Algorithm 2 according to T. Chen, L. Ljung (2013)
        aux0 = qr(hstack((transpose(Phi),Y)),mode='r')
        Rd1 = aux0[0:n+1,0:n]
        Rd2 = aux0[0:n+1,n]
        Rd2 = Rd2.reshape(len(Rd2),1)

        #cost function written as the Algorithm 2 presented in T. Chen, L. Ljung (2013)
        def algorithm2(alpha):
            L = cholesky(Prior(alpha))
            Rd1L = Rd1 @ L
            to_qr = bmat([[Rd1L,Rd2],[alpha[len(alpha)-1]*I,zeros((n,1))]])
            R = qr(to_qr,mode='r')
            R1 = R[0:n,0:n]
            r = R[n,n]
            cost = (r**2)/(alpha[len(alpha)-1]**2) + (N-n)*log(alpha[len(alpha)-1]**2) + 2*sum(log(abs(diag(R1))))
            return cost

        A = minimize(algorithm2, alpha_init, method='L-BFGS-B', bounds=bnds)
        alpha = A.x
        L = cholesky(Prior(alpha))
        Rd1L = Rd1 @ L
        to_qr = bmat([[Rd1L,Rd2],[alpha[len(alpha)-1]*I,zeros((n,1))]])
        R = qr(to_qr,mode='r')
        R1 = R[0:n,0:n]
        R2 = R[0:n,n]
        ir = L @ pinv(R1) @ R2

        ir = ir.reshape(len(ir),1)

    else:
        ir = ir_ls

    return ir

#function to check all the arguments entered by the user, raising execption if something is wrong
def argument_check(u,y,n,N,RegularizationKernel,MinimizationMethod):
    if(len(u)!=len(y)):
        raise Exception("u and y must have the same size.")

    if(n>=N):
        raise Exception("n must be at least 1 sample smaller than the length of the signals.")

    if(RegularizationKernel not in ['DC','DI','TC','SS','none']):
        raise Exception("the chosen regularization kernel is not valid.")

    if(MinimizationMethod not in ['Powell', 'TNC', 'L-BFGS-B']):
        raise Exception("the chosen minimization method is not valid. Check scipy.minimize.optimize documentation for bound-constrained methods.")

    return None