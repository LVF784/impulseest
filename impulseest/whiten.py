from numpy import prod, mean, diag, sign, sqrt, outer
from numpy.linalg import cholesky, svd

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

#function that creates the whitening matrix
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