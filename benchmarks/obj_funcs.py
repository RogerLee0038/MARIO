import numpy as np
import math
from .test_funcs import ackleyfcn, rosenbrockfcn, rastriginfcn

def ackley(x: np.ndarray) -> float:
    in_dim = len(x)
    upper_bounds = np.array([32.768] * in_dim)
    lower_bounds = np.array([-32.768] * in_dim)
    x_real = x * (upper_bounds-lower_bounds) + lower_bounds
    x_real2 = x_real[np.newaxis, : ]
    y = ackleyfcn(x_real2)[0][0]
    return y
def rosenbrock(x: np.ndarray) -> float:
    in_dim = len(x)
    upper_bounds = np.array([2.048] * in_dim)
    lower_bounds = np.array([-2.048] * in_dim)
    x_real = x * (upper_bounds-lower_bounds) + lower_bounds
    x_real2 = x_real[np.newaxis, : ]
    y = rosenbrockfcn(x_real2)[0][0]
    return y
def rastrigin(x: np.ndarray) -> float:
    in_dim = len(x)
    upper_bounds = np.array([5.12] * in_dim)
    lower_bounds = np.array([-5.12] * in_dim)
    x_real = x * (upper_bounds-lower_bounds) + lower_bounds
    x_real2 = x_real[np.newaxis, : ]
    y = rastriginfcn(x_real2)[0][0]
    return y
# for sin
def sumsin(x: np.ndarray) -> float:
    in_dim = len(x)
    upper_bounds = np.array([1] * in_dim)
    lower_bounds = np.array([-1] * in_dim)
    x_real = x * (upper_bounds-lower_bounds) + lower_bounds
    x_real2 = x_real[np.newaxis, : ]
    y = np.sin(x_real2*math.pi).sum(axis=1, keepdims=True)[0][0]
    return y