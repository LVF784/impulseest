# impulseest() is a non-parametric impulse response estimation function with input-output data

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this situation by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] using the Empirical Bayes method (Carlin and Louis [1996]).

The six arguments in this function are: <br />
    - u [NumPy array]: input signal (size N x 1); <br />
    - y [NumPy array]: output signal (size N x 1); <br />
    - n [integer]: number of impulse response estimates (default is n = 100); <br />
    - RegularizationKernel [string]: regularization method - 'none', 'DC', 'DI', 'TC' (default is 'none'); <br />
    - MinimizationMethod [string]: bound-constrained optimization method used to minimize the cost function - 'L-BFGS-B', 'Powell', 'TNC' (default is 'L-BFGS-B').

The impulseest function returns a NumPy array of size n x 1 containing all the n impulse response estimates.

## Importing

```Python
from impulseest import impulseest
```

## Example

For a detailed example, please check the example folder. Basic usage:

```Python
ir_est = impulseest(u,y,n=100,RegularizationKernel='DC')
```

## Contributor

Luan Vinícius Fiorio - vfluan@gmail.com

![Image of impulseest](https://github.com/LVF784/impulseest/blob/master/impulseest_jpeg.jpeg)
