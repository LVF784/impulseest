# impulseest() is a nonparametric impulse response estimation using only input-output data

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] using the Empirical Bayes method (Carlin and Louis [1996]).

The six arguments in this function are: <br />
    - u [numpy array]: input signal (size Nx1); <br />
    - y [numpy array]: output signal (size Nx1); <br />
    - n [int]: number of impulse response estimates (default is n=100); <br />
    - RegularizationKernel [str]: regularization method ('DC','DI','TC', default is 'none'); <br />
    - MinimizationMethod [str]: bound-constrained optimization method used to minimize the cost function ('Powell','TNC', default is 'L-BFGS-B').

The function returns a numpy array of size nx1 with all the n impulse response estimates.

## Importing impulseest function

```
from impulseest import impulseest
```

### Example of use of impulseest

```
ir_est = impulseest(u,y,n=100,RegularizationKernel='DC')
```

## Contributor

Luan Vinícius Fiorio - vfluan@gmail.com