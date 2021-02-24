# impulseest() is a nonparametric impulse response estimation with input-output data

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this situation by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] using the Empirical Bayes method (Carlin and Louis [1996]).

The six arguments in this function are: <br />
    - u [NumPy array]: input signal (size N x 1); <br />
    - y [NumPy array]: output signal (size N x 1); <br />
    - n [int]: number of impulse response estimates (default is n = 100); <br />
    - RegularizationKernel [str]: regularization method ('DC','DI','TC', default is 'none'); <br />
    - MinimizationMethod [str]: bound-constrained optimization method used to minimize the cost function ('Powell','TNC', default is 'L-BFGS-B').

The function returns a NumPy array of size n x 1 containing all the n impulse response estimates.

## Importing the function

```
from impulseest import impulseest
```

### Example of use

The impulseest() function can be used as:
```
ir_est = impulseest(u,y,n=100,RegularizationKernel='DC')
```
where u and y are, respectively, the input and output data arrays, n is the number of impulse response coefficients to be estimated, and RegularizationKernel is the method used for regularizing the estimative. Please, check the example folder for a more detailed example.

## Contributor

Luan Vinícius Fiorio - vfluan@gmail.com