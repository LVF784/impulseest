# impulseest() is a nonparametric impulse response estimation function

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] by the Empirical Bayes method (Carlin and Louis [1996]). The prewhitening filter is applied according to A. Kessy et al [2015].

The six arguments in this function are: <br />
    - u [numpy array]: input signal (size Nx1); <br />
    - y [numpy array]: output signal (size Nx1); <br />
    - n [int]: number of impulse response estimates (default is n=100); <br />
    - RegularizationKernel [str]: regularization method ('DC','DI','TC', default is 'none'); <br />
    - PreFilter [str]: prewhitening filter method ('zca', 'pca', 'cholesky', 'pca_cor', 'zca_cor', default is 'none'); <br />
    - MinimizationMethod [str]: bound-constrained optimization method used to minimize the cost function ('Powell','TNC', default is 'L-BFGS-B').

## Importing function

```
from impulseest import impulseest
```

## Contributor

Luan Vinícius Fiorio - vfluan@gmail.com