import numpy as np

def ackleyfcn(x: np.ndarray) -> np.ndarray:
    """
    Computes the value of Ackley benchmark function.

    Parameters:
    x (numpy.ndarray): Input matrix of size M-by-N.

    Returns:
    numpy.ndarray: Vector SCORES of size M-by-1 in which each row contains the function value for 
    each row of X.

    Ackley function is a benchmark function for optimization problems.
    For more information please visit:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Author: Mazhar Ansari Ardeh
    Please forward any comments or bug reports to mazhar.ansari.ardeh at
    Google's e-mail service or feel free to kindly modify the repository.
    """
    n = x.shape[1]
    ninverse = 1 / n
    sum1 = np.sum(x**2, axis=1, keepdims=True)
    sum2 = np.sum(np.cos(2 * np.pi * x), axis=1, keepdims=True)

    scores = (
        20
        + np.exp(1)
        - (20 * np.exp(-0.2 * np.sqrt(ninverse * sum1)))
        - np.exp(ninverse * sum2)
    )
    return scores

def rosenbrockfcn(x: np.ndarray) -> np.ndarray:
    """
    Computes the value of the Rosenbrock benchmark function.

    Parameters
    ----------
    x : numpy.ndarray of shape (M, N)
        The input matrix where each row is a sample of N-dimensional column vector.

    Returns
    -------
    numpy.ndarray of shape (M, 1)
        A vector containing the function value for each row of x.

    For more information please visit:
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Author: Mazhar Ansari Ardeh
    Please forward any comments or bug reports to mazhar.ansari.ardeh at
    Google's e-mail service or feel free to kindly modify the repository.
    """
    scores = np.zeros(x.shape[0])
    n = x.shape[1]
    assert n >= 1, "Given input X cannot be empty"
    a = 1
    b = 100
    for i in range(n - 1):
        scores += b * ((x[:, i + 1] - (x[:, i] ** 2)) ** 2) + (
           (a - x[:, i]) ** 2
         )
    return scores[:, np.newaxis]

def rastriginfcn(x: np.ndarray) -> np.ndarray:
    """
    Computes the value of Rastrigin benchmark function.

    Parameters
    ----------
    x : numpy.ndarray
        Input matrix of size M-by-N.

    Returns
    -------
    numpy.ndarray
        Output vector SCORES of size M-by-1.

    For more information, please visit:
    https://en.wikipedia.org/wiki/Rastrigin_function

    Author: Mazhar Ansari Ardeh

    Please forward any comments or bug reports to mazhar.ansari.ardeh at
    Google's e-mail service or feel free to kindly modify the repository.
    """

    n = x.shape[1]
    A = 10
    f = (A * n) + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)
    return f.reshape(-1, 1)