impulseest() is a nonparametric impulse response estimation function.

As the variance increases linearly with the finite impulse response (FIR) model order, it is important for higher order FIR models to counteract this by regularizing the estimative. In impulseest(), this is done as proposed in T. Chen et al [2012] by the Empirical Bayes method (Carlin and Louis [1996]).

The function has four arguments: input data, output data, number of impulse response estimates and regularization kernel.
- each of input and output data arrays must be an N x 1 or 1 x N array. Both must have the same length; 
- the number of impulse response estimates is chosen arbitrarly, with default value of n = 100; 
- there are four regularization kernels to be chosen ('DC', 'DI', 'TC', 'CS') and a 'none' option. If you choose 'none', the function returns the impulse response estimated by least squares with no regularization. The default choice is 'CS'.
