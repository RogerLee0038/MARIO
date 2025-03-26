from nevergrad.functions import ArtificialFunction
from nevergrad.functions.corefuncs import registry 
import numpy as np

@registry.register
def schwefel(x: np.ndarray) -> float:
    print("x", x)
    dim = x.size
    res = 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    print("res", res)
    return res

@registry.register
def langermann(x: np.ndarray) -> float:
    assert x.size == 2
    c = np.array([1, 2, 5, 2, 3])
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    m = 5
    res = np.sum(
        [
            c[i]
            * np.exp(-1 / np.pi * np.sum((x - A[i]) ** 2))
            * np.cos(np.pi * np.sum((x - A[i]) ** 2))
            for i in range(m)
        ]
    )
    return res

@registry.register
def michalewicz(x: np.ndarray) -> float:
    m = 10
    dim = x.size
    i = np.arange(1, dim + 1)
    res = -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m))
    return res

class MyArtificialFunction(ArtificialFunction):
    def __init__(
        self,
        name: str,
        block_dimension: int,
        num_blocks: int = 1,
        useless_variables: int = 0,
        noise_level: float = 0,
        noise_dissymmetry: bool = False,
        rotation: bool = False,
        translation_factor: float = 1.0,
        hashing: bool = False,
        aggregator: str = "max",
        split: bool = False,
        bounded: bool = False,
        expo: float = 1.0,
        zero_pen: bool = False,
    ) -> None:
        super().__init__(
            name,
            block_dimension,
            num_blocks,
            useless_variables,
            noise_level,
            noise_dissymmetry,
            rotation,
            translation_factor,
            hashing,
            aggregator,
            split,
            bounded,
            expo,
            zero_pen,
        )
        if name == "ackley": # 0 at 0*in_dim
            self.bounds_lower = -32.768
            self.bounds_upper = 32.768
        elif name == "rastrigin": # 0 at 0*in_dim
            self.bounds_lower = -5.12
            self.bounds_upper = 5.12
        elif name == "rosenbrock": # 0 at 1*in_dim
            self.bounds_lower = -2.048
            self.bounds_upper = 2.048
        elif name == "griewank": # 0 at 0*in_dim
            self.bounds_lower = -600
            self.bounds_upper = 600
        elif name == "schwefel": # 0 at 420.9687*in_dim
            self.bounds_lower = -500
            self.bounds_upper = 500
        elif name == "langermann":
            self.bounds_lower = 0
            self.bounds_upper = 10
        elif name == "michalewicz":
            self.bounds_lower = 0
            self.bounds_upper = np.pi
        else: # lunacek, multipeak, deceptivemultimodal, hm, bucherastrigin
            self.bounds_lower = -5
            self.bounds_upper = 5