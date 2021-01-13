# impulseest() is a nonparametric impulse response estimation function

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] by the Empirical Bayes method (Carlin and Louis [1996]). The prewhitening filter is applied according to A. Kessy et al [2015].

The six arguments in this function are: <br />
    - u [numpy array]: input signal (size Nx1); <br />
    - y [numpy array]: output signal (size Nx1); <br />
    - n [int]: number of impulse response estimates (default is n=100); <br />
    - RegularizationKernel [str]: regularization method ('DC','DI','TC', default is 'none'); <br />
    - PreFilter [str]: prewhitening filter method ('zca', 'pca', 'cholesky', 'pca_cor', 'zca_cor', default is 'none'); <br />
    - MinimizationMethod [str]: bound-constrained optimization method used to minimize the cost function ('Powell','TNC', default is 'L-BFGS-B').

The function returns a numpy array of size nx1 with all the n impulse response estimates.

## Importing impulseest function

```
from impulseest import impulseest
```

### Example of use of impulseest

```
ir_est = impulseest(u,y,n=100,RegularizationKernel='DI')
```

## Whitening filter

The whitening filter can be used separately from impulseest main function, by importing the forementioned as follows:

```
from impulseest import whiten
```

A signal can be whitened according to the theory presented in A. Kessy et al [2015] with the function whiten. The arguments of this function are:<br />
    - x [numpy array]: discrete-time signal (size Nx1);<br />
    - method [str]: method used to create the W matrix, available options
    are: 'zca', 'pca', 'cholesky', 'zca_cor', 'pca_cor'.

The whiten function returns a whitened discrete-time signal of the same size as the input array x.

### Example of use of whiten

```
u_whitened = whiten(u)
```

## Contributor

Luan Vinícius Fiorio - vfluan@gmail.com