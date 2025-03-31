import sys
from numpy import abs, cos, sin, tan, arctan, exp, log, pi, prod, sqrt, sum, asarray, arange, atleast_2d
from .go_benchmark import Benchmark


class MyAckley(Benchmark):
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-30.0] * self.N, [30.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        u = sum(x ** 2)
        v = sum(cos(2 * pi * x))
        return (-20. * exp(-0.02 * sqrt(u / self.N))
                - exp(v / self.N) + 20. + exp(1.))


class MyAluffiPentini(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[-1.0465, 0]]
        self.fglob = -0.3523861

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 0.25*x[0]**4 - 0.5*x[0]**2 + 0.1*x[0] + 0.5*x[1]**2


class MyBeckerLago(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[5, 5],[5, -5],[-5, 5],[-5, -5]]
        self.fglob = 0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return (abs(x[0]-5))**2 + (abs(x[1]-5))**2


class MyBohachevsky1(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0])
                - 0.4 * cos(4 * pi * x[1]) + 0.7)


class MyBohachevsky2(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0])
                 * cos(4 * pi * x[1]) + 0.3)


class MyBranin(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(-5., 10.), (0., 15.)]
        self.global_optimum = [[-pi, 12.275], [pi, 2.275], [3 * pi, 2.475]]
        self.fglob = 0.39788735772973816

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return ((x[1] - (5.1 / (4 * pi ** 2)) * x[0] ** 2
                + 5 * x[0] / pi - 6) ** 2
                + 10 * (1 - 1 / (8 * pi)) * cos(x[0]) + 10)


class MyCamel3(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(-5., 5.), (-5., 5.)]
        self.global_optimum = [[0,0]]
        self.fglob = 0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 2*x[0]**2 - 1.05*x[0]**4 + (1/6)*x[0]**6 + x[0]*x[1] + x[1]**2


class MyCamel6(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(-5., 5.), (-5., 5.)]
        self.global_optimum = [[0.089842,-0.712656],[-0.089842,0.712656]]
        self.fglob = -1.0316

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4


class MyCosineMixture(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self.change_dimensionality = True
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0. for _ in range(self.N)]]
        self.fglob = -0.1 * self.N

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -0.1 * sum(cos(5.0 * pi * x)) + sum(x ** 2.0)


class MyDekkersAarts(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(-20, 20),(-20,20)]
        self.global_optimum = [[0,15],[0,-15]]
        self.fglob = -24777

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 10**5*x[0]**2 + x[1]**1 -(x[0]**2+x[1]**2)**2 + 10**(-5)*(x[0]**2 + x[1]**2)**4


class MyEasom(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N,
                           [10.0] * self.N))

        self.global_optimum = [[pi for _ in range(self.N)]]
        self.fglob = -1.0

    def fun(self, x, *args):
        self.nfev += 1
        a = (x[0] - pi)**2 + (x[1] - pi)**2

        return -cos(x[0]) * cos(x[1]) * exp(-a)


class MyMichalewicz2(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))
        self.global_optimum = [[2.20290555, 1.570796]]
        self.fglob = -1.8013

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        m = 10.0
        i = arange(1, self.N + 1)
        return -sum(sin(x) * sin(i * x ** 2 / pi) ** (2 * m))


class MyMichalewicz5(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))
        self.global_optimum = [[2.693, 0.259, 2.074, 1.023, 1.720]]
        # If no transform
        # self.global_optimum = [[2.202906, 1.570796, 1.284992, 1.923058, 1.720470]]
        self.fglob = -4.6876582

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        m = 10.0
        i = arange(1, self.N + 1)
        # Transform
        y = x.copy()
        theta = pi/6
        for k in range(len(y)-1): # N is odd
            if not k%2:
                y[k] = x[k]*cos(theta)-x[k+1]*sin(theta)
            else:
                y[k] = x[k]*cos(theta)+x[k-1]*sin(theta)
        return -sum(sin(y) * sin(i * y ** 2 / pi) ** (2 * m))


class MyMichalewicz10(Benchmark):
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))
        self.global_optimum = [[2.693, 0.259, 2.074, 1.023, 2.275, 0.500, 2.138, 0.794, 2.219, 0.533]] 
        # If no transform
        #self.global_optimum = [
        #    [2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 
        #     1.570796, 1.454414, 1.756087, 1.655717, 1.570796]
        #]
        self.fglob = -9.6601517

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        m = 10.0
        i = arange(1, self.N + 1)
        # Transform
        y = x.copy()
        theta = pi/6
        for k in range(len(y)): # N is not odd
            if not k%2:
                y[k] = x[k]*cos(theta)-x[k+1]*sin(theta)
            else:
                y[k] = x[k]*cos(theta)+x[k-1]*sin(theta)
        print("y", y)
        return -sum(sin(y) * sin(i * y ** 2 / pi) ** (2 * m))


class MyExponential(Benchmark):
    def __init__(self, dimensions=40):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = -1.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -exp(-0.5 * sum(x ** 2.0))
        

class MyGoldsteinPrice(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        self.global_optimum = [[0., -1.]]
        self.fglob = 3.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        a = (1 + (x[0] + x[1] + 1) ** 2
             * (19 - 14 * x[0] + 3 * x[0] ** 2
             - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))
        b = (30 + (2 * x[0] - 3 * x[1]) ** 2
             * (18 - 32 * x[0] + 12 * x[0] ** 2
             + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
        return a * b
        

class MyGriewank(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-600.0] * self.N,
                           [600.0] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        i = arange(1., len(x) + 1.)
        return sum(x ** 2 / 4000) - prod(cos(x / sqrt(i))) + 1



class MyGulf(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.1, 0.0, 0.0], [100.0, 25.6, 5]))
        self.global_optimum = [[50.0, 25.0, 1.5]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        m = 99.
        i = arange(1., m + 1)
        u = 25 + (-50 * log(i / 100.)) ** (2 / 3.)
        vec = (exp(-((abs(u - x[1])) ** x[2] / x[0])) - i / 100.)
        return sum(vec ** 2)


class MyHartman3(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))
        self.global_optimum = [[0.11461292, 0.55564907, 0.85254697]]
        self.fglob = -3.8627821478

        self.a = asarray([[3.0, 10., 30.],
                          [0.1, 10., 35.],
                          [3.0, 10., 30.],
                          [0.1, 10., 35.]])

        self.p = asarray([[0.3689, 0.1170, 0.2673],
                          [0.4699, 0.4387, 0.7470],
                          [0.1091, 0.8732, 0.5547],
                          [0.03815, 0.5743, 0.8828]])

        self.c = asarray([1., 1.2, 3., 3.2])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        XX = atleast_2d(x)
        d = sum(self.a * (XX - self.p) ** 2, axis=1)
        return -sum(self.c * exp(-d))



class MyHartman6(Benchmark):
    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        self.global_optimum = [[0.20168952, 0.15001069, 0.47687398, 0.27533243,
                                0.31165162, 0.65730054]]

        self.fglob = -3.32236801141551

        self.a = asarray([[10., 3., 17., 3.5, 1.7, 8.],
                          [0.05, 10., 17., 0.1, 8., 14.],
                          [3., 3.5, 1.7, 10., 17., 8.],
                          [17., 8., 0.05, 10., 0.1, 14.]])

        self.p = asarray([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                          [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                          [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                          [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        self.c = asarray([1.0, 1.2, 3.0, 3.2])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        XX = atleast_2d(x)
        d = sum(self.a * (XX - self.p) ** 2, axis=1)
        return -sum(self.c * exp(-d))
        

class MyHelicalValley(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.] * self.N, [10.] * self.N))
        self.global_optimum = [[1.0, 0.0, 0.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        r = sqrt(x[0] ** 2 + x[1] ** 2)
        if x[0] >= 0:
            theta = 1 / (2. * pi) * arctan(x[1]/x[0])
        else:
            theta = 1 / (2. * pi) * arctan(x[1]/x[0] + 0.5)

        return x[2] ** 2 + 100 * ((x[2] - 10 * theta) ** 2 + (r - 1) ** 2)


class MyHosaki(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = ([0., 5.], [0., 6.])
        self.global_optimum = [[4, 2]]
        self.fglob = -2.3458115

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        val = (1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3. * x[0] ** 3
               + 0.25 * x[0] ** 4)
        return val * x[1] ** 2 * exp(-x[1])


class MyKowalik(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.] * self.N, [0.42] * self.N))
        self.global_optimum = [[0.192833, 0.190836, 0.123117, 0.135766]]
        self.fglob = 0.00030748610

        self.a = asarray([4.0, 2.0, 1.0, 1 / 2.0, 1 / 4.0, 1 / 6.0, 1 / 8.0,
                          1 / 10.0, 1 / 12.0, 1 / 14.0, 1 / 16.0])
        self.b = asarray([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                          0.0456, 0.0342, 0.0323, 0.0235, 0.0246])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        vec = self.b - (x[0] * (self.a ** 2 + self.a * x[1])
                   / (self.a ** 2 + self.a * x[2] + x[3]))
        return sum(vec ** 2)


class MyLevyMontalvo1(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[-1 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        y = 1 + (x + 1) / 4
        v = sum((y[:-1] - 1) ** 2 * (1 + 10 * sin(pi * y[1:]) ** 2))
        z = (y[-1] - 1) ** 2
        return pi/self.N * (10 * sin(pi * y[0]) ** 2 + v + z)


class MyLevyMontalvo2(Benchmark):
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        v = sum((x[:-1] - 1) ** 2 * (1 + sin(3 * pi * x[1:]) ** 2))
        z = (x[-1] - 1) ** 2 * (1 + sin(2 * pi * x[-1])**2)
        return 0.1 * (sin(3 * pi * x[0]) ** 2 + v + z)


class MyMcCormick(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(-1.5, 4.0), (-3.0, 3.0)]

        self.global_optimum = [[-0.5471975602214493, -1.547197559268372]]
        self.fglob = -1.913222954981037

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return (sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0]
                + 2.5 * x[1] + 1)


class MyMeyerRoth(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-20.] * self.N, [20] * self.N))
        self.global_optimum = [[3.13, 15.16, 0.78]]
        self.fglob = 4.355269e-5

        self.t = asarray([1.0, 2.0, 1.0, 2.0, 0.1])
        self.v = asarray([1.0, 1.0, 2.0, 2.0, 0.0])
        self.y = asarray([0.126, 0.219, 0.076, 0.126, 0.186])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        vec = x[0] * x[2] * self.t / (1 + x[0] * self.t + x[1] * self.v)
        return sum((vec - self.y) ** 2)


class MyMieleCantrell(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-1.] * self.N, [1.] * self.N))
        self.global_optimum = [[0., 1., 1., 1.]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return (exp(x[0]) - x[1]) ** 4 + 100 * (x[1] - x[2]) ** 6 + tan(x[2] - x[3]) ** 4 + x[0] ** 8


class MyModifiedLangerman5(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[8.074, 8.777, 3.467, 1.867, 6.708]]
        self.fglob = -0.965

        self.c = asarray([0.806, 0.517, 0.100, 0.908, 0.965])
        self.a = asarray(
            [[9.681, 9.400, 8.025, 2.196, 8.074],
             [0.667, 2.041, 9.152, 0.415, 8.777],
             [4.783, 3.788, 5.114, 5.649, 3.467],
             [9.095, 7.931, 7.621, 6.979, 1.867],
             [3.517, 2.882, 4.564, 9.510, 6.708]]
        ).T

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        XX = atleast_2d(x)
        d = sum((XX - self.a) ** 2, axis = 1)
        return -sum(self.c * cos(d / pi) * exp(-pi * d))


class MyModifiedLangerman10(Benchmark):
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[8.074, 8.777, 3.467, 1.867, 6.708,
                                6.349, 4.534, 0.276, 7.633, 1.567]]
        self.fglob = -0.965

        self.c = asarray([0.806, 0.517, 0.100, 0.908, 0.965])
        self.a = asarray(
            [[9.681, 9.400, 8.025, 2.196, 8.074],
             [0.667, 2.041, 9.152, 0.415, 8.777],
             [4.783, 3.788, 5.114, 5.649, 3.467],
             [9.095, 7.931, 7.621, 6.979, 1.867],
             [3.517, 2.882, 4.564, 9.510, 6.708],
             [9.325, 2.672, 4.711, 9.166, 6.349],
             [6.544, 3.568, 2.996, 6.304, 4.534],
             [0.211, 1.284, 6.126, 6.054, 0.276],
             [5.122, 7.033, 0.734, 9.377, 7.633],
             [2.020, 7.374, 4.982, 1.426, 1.567],]
        ).T

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        XX = atleast_2d(x)
        d = sum((XX - self.a) ** 2, axis = 1)
        return -sum(self.c * cos(d / pi) * exp(-pi * d))


class MyModifiedRosenbrock(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.] * self.N, [5.] * self.N))
        self.global_optimum = [[1 for _ in range(self.N)], [0.3412, 0.1164]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 100 * (x[1] - x[0] ** 2) ** 2 + (6.4 * (x[1] - 0.5) ** 2 - x[0] - 0.6) ** 2


class MyMultiGaussian(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-2.] * self.N, [2.] * self.N))
        self.global_optimum = [[-0.01356, -0.01356]]
        self.fglob = -1.29695

        self.a = asarray([0.5, 1.2, 1.0, 1.0, 1.2])
        self.b = asarray([0.0, 1.0, 0.0, -0.5, 0.0])
        self.c = asarray([0.0, 0.0, -0.5, 0.0, 1.0])
        self.d = asarray([0.1, 0.5, 0.5, 0.5, 0.5])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(self.a * exp(-((x[0] - self.b) ** 2 + (x[1] - self.c) ** 2) / self.d ** 2))


class MyNeumaier2(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.] * self.N, [self.N] * self.N))
        self.global_optimum = [[1, 2, 2, 3]]
        self.fglob = 0.0

        self.b = asarray([8, 18, 44, 114])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        v = []
        for i in arange(1, self.N + 1):
            v.append(sum(x ** i))
        v = asarray(v)
        return sum((self.b - v) ** 2)


class MyNeumaier3(Benchmark):
    def __init__(self, dimensions=30):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-(self.N) ** 2] * self.N, [(self.N) ** 2] * self.N))
        self.global_optimum = [[i * (self.N + 1 - i) for i in range(1, self.N + 1)]]
        self.fglob = -self.N * (self.N + 4) * (self.N - 1) / 6
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        v = []
        for i in arange(1, self.N):
            v.append(x[i] * x[i-1])
        v = asarray(v)
        return sum((x - 1) ** 2) - sum(v)


class MyOddSquare(Benchmark):
    def __init__(self, dimensions=20):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-15.0] * self.N,
                           [15.0] * self.N))
        self.b = asarray([1, 1.3, 0.8, -0.4, -1.3, 1.6, -2, -6, 0.5, 1.4,
                          1, 1.3, 0.8, -4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4])
        self.global_optimum = [[1, 1.3, 0.8, -0.4, -1.3, 1.6, -2, -6, 0.5, 1.4,
                                1, 1.3, 0.8, -4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4]]
        self.fglob = -1.

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        d = sqrt(sum((x - self.b) ** 2.0))
        D = sqrt(self.N) * max(abs(x - self.b))
        return (-exp(-D / (2.0 * pi)) * cos(pi * D)
                * (1.0 + 0.2 * d / (D + 0.1)))


class MyPaviani(Benchmark):
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([2.001] * self.N, [9.999] * self.N))
        self.global_optimum = [[9.350266 for _ in range(self.N)]]
        self.fglob = -45.7784684040686
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return sum(log(x - 2) ** 2.0 + log(10.0 - x) ** 2.0) - prod(x) ** 0.2


class MyPeriodic(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10., -10.], [10., 10.]))
        self.global_optimum = [[0., 0.]]
        self.fglob = 0.9

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 1 + sin(x[0]) ** 2 + sin(x[1]) ** 2 - 0.1 * exp(-(x[0] ** 2 + x[1] ** 2))


class MyPowellQuadratic(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.] * self.N, [10.] * self.N))
        self.global_optimum = [[0.] * self.N]
        self.fglob = 0.

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return (11 * x[0]) ** 2 + 5 * (x[2] - x[3]) ** 2 \
               + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4


class MyPriceTransistor(Benchmark):
    def __init__(self, dimensions=9):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.] * self.N, [10.] * self.N))
        self.global_optimum = [[0.9, 0.45, 1, 2, 8, 8, 5, 1, 2]]
        self.fglob = 0.

        self.g = asarray(
            [[0.485, 0.752, 0.869, 0.982],
             [0.369, 1.254, 0.703, 1.455],
             [5.2095, 10.0677, 22.9274, 20.2153],
             [23.3037, 101.779, 111.461, 191.267],
             [28.5132, 111.8467, 134.3884, 211.4823]]
        )

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        alpha = (1 - x[0] * x[1]) * x[2] \
        * (exp(x[4] * (self.g[0] - self.g[2] * x[6] * 1e-3 - self.g[4] * x[7] * 1e-3)) - 1) \
        - self.g[4] + self.g[3] * x[1]
        beta = (1 - x[0] * x[1]) * x[3] \
        * (exp(x[5] * (self.g[0] - self.g[1] - self.g[2] * x[6] * 1e-3 + self.g[3] * x[8] * 1e-3)) - 1) \
        - self.g[4] * x[0] + self.g[3]
        gamma = x[0] * x[2] - x[1] * x[3]
        return gamma ** 2 + sum(alpha ** 2 + beta ** 2)


class MyRastrigin(Benchmark):
    def __init__(self, dimensions=30):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.12] * self.N, [5.12] * self.N))
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return 10.0 * self.N + sum(x ** 2.0 - 10.0 * cos(2.0 * pi * x))


class MyRosenbrock(Benchmark): 
    def __init__(self, dimensions=50):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-30.] * self.N, [30.0] * self.N))
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        v = []
        u = []
        for i in range(self.N-1):
            v.append(x[i+1] - x[i] ** 2)
            u.append((x[i] - 1) ** 2)
        v = asarray(v)
        u = asarray(u)
        return sum(100 * v ** 2 + u)


class MySalomon(Benchmark):
    def __init__(self, dimensions=50):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        u = sqrt(sum(x ** 2))
        return 1 - cos(2 * pi * u) + 0.1 * u


class MySchaffer1(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        u = (x[0] ** 2 + x[1] ** 2)
        num = sin(sqrt(u)) ** 2 - 0.5
        den = (1 + 0.001 * u) ** 2
        return 0.5 + num / den


class MySchaffer2(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        r2 = x[0] ** 2 + x[1] ** 2
        return r2 ** 0.25 * (sin(50 * r2 ** 0.1) ** 2 + 1)


class MySchwefel(Benchmark):
    def __init__(self, dimensions=40):
        Benchmark.__init__(self, dimensions)
        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))

        self.global_optimum = [[420.968746 for _ in range(self.N)]]
        self.fglob = -418.982887 * self.N
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(x * sin(sqrt(abs(x))))


class MyShekel5(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[4.00003715092,
                                4.00013327435,
                                4.00003714871,
                                4.0001332742]]
        self.fglob = -10.1531996791

        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0]])
        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(1 / (sum((x - self.A) ** 2, axis=1) + self.C))


class MyShekel7(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[4.00057291078,
                                4.0006893679,
                                3.99948971076,
                                3.99960615785]]
        self.fglob = -10.4029405668

        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0],
                          [2.0, 9.0, 2.0, 9.0],
                          [5.0, 5.0, 3.0, 3.0]])
        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(1 / (sum((x - self.A) ** 2, axis=1) + self.C))


class MyShekel10(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[4.0007465377266271,
                                4.0005929234621407,
                                3.9996633941680968,
                                3.9995098017834123]]
        self.fglob = -10.536409816692023

        self.A = asarray([[4.0, 4.0, 4.0, 4.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [8.0, 8.0, 8.0, 8.0],
                          [6.0, 6.0, 6.0, 6.0],
                          [3.0, 7.0, 3.0, 7.0],
                          [2.0, 9.0, 2.0, 9.0],
                          [5.0, 5.0, 3.0, 3.0],
                          [8.0, 1.0, 8.0, 1.0],
                          [6.0, 2.0, 6.0, 2.0],
                          [7.0, 3.6, 7.0, 3.6]])
        self.C = asarray([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(1 / (sum((x - self.A) ** 2, axis=1) + self.C))


class MyShekelFoxholes5(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[8.025, 9.152, 5.114, 7.621, 4.564]]
        self.fglob = -10.4056

        self.P = asarray(
            [[0.806, 9.681, 0.667, 4.783, 9.095, 3.517],
             [0.517, 9.400, 2.041, 3.788, 7.931, 2.882],
             [0.100, 8.025, 9.152, 5.114, 7.621, 4.564],
             [0.908, 2.196, 0.415, 5.649, 6.979, 9.510],
             [0.965, 8.074, 8.777, 3.467, 1.863, 6.708],
             [0.669, 7.650, 5.658, 0.720, 2.764, 3.278],
             [0.524, 1.256, 3.605, 8.623, 6.905, 4.584],
             [0.902, 8.314, 2.261, 4.224, 1.781, 4.124],
             [0.531, 0.226, 8.858, 1.420, 0.945, 1.622],
             [0.876, 7.305, 2.228, 1.242, 5.928, 9.133],
             [0.462, 0.652, 7.027, 0.508, 4.876, 8.807],
             [0.491, 2.699, 3.516, 5.874, 4.119, 4.461],
             [0.463, 8.327, 3.897, 2.017, 9.570, 9.825],
             [0.714, 2.132, 7.006, 7.136, 2.641, 1.882],
             [0.352, 4.707, 5.579, 4.080, 0.581, 9.698],
             [0.869, 8.304, 7.559, 8.567, 0.322, 7.128],
             [0.813, 8.632, 4.409, 4.832, 5.768, 7.050],
             [0.811, 4.887, 9.112, 0.170, 8.967, 9.693],
             [0.828, 2.440, 6.686, 4.299, 1.007, 7.008],
             [0.964, 6.306, 8.583, 6.084, 1.138, 4.350],
             [0.789, 0.652, 2.343, 1.370, 0.821, 1.310],
             [0.360, 5.558, 1.272, 5.756, 9.857, 2.279],
             [0.369, 3.352, 7.549, 9.817, 9.437, 8.687],
             [0.992, 8.798, 0.880, 2.370, 0.168, 1.701],
             [0.332, 1.460, 8.057, 1.336, 7.217, 7.914],
             [0.817, 0.432, 8.645, 8.774, 0.249, 8.081],
             [0.632, 0.679, 2.800, 5.523, 3.049, 2.968],
             [0.883, 4.263, 1.074, 7.286, 5.599, 8.291],
             [0.608, 9.496, 4.830, 3.150, 8.270, 5.079],
             [0.326, 4.138, 2.562, 2.532, 9.661, 5.611]]
        )
        self.c = self.P[:, 0]
        self.a = self.P[:, 1:]

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(1 / (sum((x - self.a) ** 2, axis=1) + self.c))


class MyShekelFoxholes10(Benchmark):
    def __init__(self, dimensions=10):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        # self.global_optimum = [[8.025, 9.152, 5.114, 7.621, 4.564,
        #                         4.771, 2.996, 6.126, 0.734, 4.982]] # seems wrong
        self.global_optimum = [[8.024961, 9.151914, 5.113990, 7.620956, 4.564025, 
                                4.710999, 2.996038, 6.125996, 0.734065, 4.981999]]
        self.fglob = -10.2088

        self.P = asarray(
            [[0.806, 9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020],
             [0.517, 9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374],
             [0.100, 8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982],
             [0.908, 2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426],
             [0.965, 8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567],
             [0.669, 7.650, 5.658, 0.720, 2.764, 3.278, 5.283, 7.474, 6.274, 1.409, 8.208],
             [0.524, 1.256, 3.605, 8.623, 6.905, 4.584, 8.133, 6.071, 6.888, 4.187, 5.448],
             [0.902, 8.314, 2.261, 4.224, 1.781, 4.124, 0.932, 8.129, 8.658, 1.208, 5.762],
             [0.531, 0.226, 8.858, 1.420, 0.945, 1.622, 4.698, 6.228, 9.096, 0.972, 7.637],
             [0.876, 7.305, 2.228, 1.242, 5.928, 9.133, 1.826, 4.060, 5.204, 8.713, 8.247],
             [0.462, 0.652, 7.027, 0.508, 4.876, 8.807, 4.632, 5.808, 6.937, 3.291, 7.016],
             [0.491, 2.699, 3.516, 5.874, 4.119, 4.461, 7.496, 8.817, 0.690, 6.593, 9.789],
             [0.463, 8.327, 3.897, 2.017, 9.570, 9.825, 1.150, 1.395, 3.885, 6.354, 0.109],
             [0.714, 2.132, 7.006, 7.136, 2.641, 1.882, 5.943, 7.273, 7.691, 2.880, 0.564],
             [0.352, 4.707, 5.579, 4.080, 0.581, 9.698, 8.542, 8.077, 8.515, 9.231, 4.670],
             [0.869, 8.304, 7.559, 8.567, 0.322, 7.128, 8.392, 1.472, 8.524, 2.277, 7.826],
             [0.813, 8.632, 4.409, 4.832, 5.768, 7.050, 6.715, 1.711, 4.323, 4.405, 4.591],
             [0.811, 4.887, 9.112, 0.170, 8.967, 9.693, 9.867, 7.508, 7.770, 8.382, 6.740],
             [0.828, 2.440, 6.686, 4.299, 1.007, 7.008, 1.427, 9.398, 8.480, 9.950, 1.675],
             [0.964, 6.306, 8.583, 6.084, 1.138, 4.350, 3.134, 7.853, 6.061, 7.457, 2.258],
             [0.789, 0.652, 2.343, 1.370, 0.821, 1.310, 1.063, 0.689, 8.819, 8.833, 9.070],
             [0.360, 5.558, 1.272, 5.756, 9.857, 2.279, 2.764, 1.284, 1.677, 1.244, 1.234],
             [0.369, 3.352, 7.549, 9.817, 9.437, 8.687, 4.167, 2.570, 6.540, 0.228, 0.027],
             [0.992, 8.798, 0.880, 2.370, 0.168, 1.701, 3.680, 1.231, 2.390, 2.499, 0.064],
             [0.332, 1.460, 8.057, 1.336, 7.217, 7.914, 3.615, 9.981, 9.198, 5.292, 1.224],
             [0.817, 0.432, 8.645, 8.774, 0.249, 8.081, 7.461, 4.416, 0.652, 4.002, 4.644],
             [0.632, 0.679, 2.800, 5.523, 3.049, 2.968, 7.225, 6.730, 4.199, 9.614, 9.229],
             [0.883, 4.263, 1.074, 7.286, 5.599, 8.291, 5.200, 9.214, 8.272, 4.398, 4.506],
             [0.608, 9.496, 4.830, 3.150, 8.270, 5.079, 1.231, 5.731, 9.494, 1.883, 9.732],
             [0.326, 4.138, 2.562, 2.532, 9.661, 5.611, 5.500, 6.886, 2.341, 9.699, 6.500]]
        )
        self.c = self.P[:, 0]
        self.a = self.P[:, 1:]

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -sum(1 / (sum((x - self.a) ** 2, axis=1) + self.c))


class MyShubert(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[-7.0835, 4.8580]] # and more
        self.fglob = -186.7309

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        j = atleast_2d(arange(1, 6)).T
        y = j * cos((j + 1) * x + j)
        return prod(sum(y, axis=0))


class MySinusoidal(Benchmark):
    def __init__(self, dimensions=20):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.] * self.N, [180.] * self.N))
        self.global_optimum = [[120.] * self.N] # and more
        self.fglob = -3.5
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)
        return -(2.5 * prod(sin((x - 30) / 180 * pi)) + prod(sin(5 * ((x - 30) / 180 * pi))))


class MyStornTchebyshev9(Benchmark):
    def __init__(self, dimensions=9):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-256.] * self.N, [256.] * self.N))
        self.global_optimum = [[128, 0, -256, 0, 160, 0, -32, 0, 1]]
        self.fglob = 0.0
        self.d = 72.661
        self.m = 60

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        i = arange(1, self.N + 1)
        u = sum((1.2) ** (self.N - i) * x[i-1])
        v = sum((-1.2) ** (self.N - i) * x[i-1])
        w = [sum((2 * j / self.m - 1) ** (self.N - i) * x[i-1]) for j in range(self.m)]
        pj = []
        for j in range(self.m):
            if w[j] > 1:
                pj.append((w[j] - 1) ** 2)
            elif w[j] < -1:
                pj.append((w[j] + 1) ** 2)
            else:
                pj.append(0)
        pj = asarray(pj)

        p1 = (u - self.d) ** 2 if u < self.d else 0
        p2 = (v - self.d) ** 2 if v < self.d else 0
        p3 = sum(pj)
        return p1 + p2 + p3


class MyStornTchebyshev17(Benchmark):
    def __init__(self, dimensions=17):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-32768.] * self.N, [32768.] * self.N))
        self.global_optimum = [[32768, 0, -131072, 0, 212992, 0, -180224, 0, 84480, 0, -21504, 0, 2688, 0, -128, 0, 1]]
        self.fglob = 0.0
        self.d = 10558.145
        self.m = 100

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        i = arange(1, self.N + 1)
        u = sum((1.2) ** (self.N - i) * x[i-1])
        v = sum((-1.2) ** (self.N - i) * x[i-1])
        w = [sum((2 * j / self.m - 1) ** (self.N - i) * x[i-1]) for j in range(self.m)]
        pj = []
        for j in range(self.m):
            if w[j] > 1:
                pj.append((w[j] - 1) ** 2)
            elif w[j] < -1:
                pj.append((w[j] + 1) ** 2)
            else:
                pj.append(0)
        pj = asarray(pj)

        p1 = (u - self.d) ** 2 if u < self.d else 0
        p2 = (v - self.d) ** 2 if v < self.d else 0
        p3 = sum(pj)
        return p1 + p2 + p3


class MyWood(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.global_optimum = [[1, 1, 1, 1]]
        self.fglob = 0

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 \
               + 90 * (x[3] - x[2] ** 2) ** 2 + (1 - x[2]) ** 2 \
               + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) \
               + 19.8 * (x[1] - 1) * (x[3] - 1)


class MyPutativeMichalewicz(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))
        # 5_dim check
        self.global_optimum = [[2.202906, 1.570796, 1.284992, 1.923058, 1.720470]]
        self.fglob = -0.99864 * self.N + 0.30271
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        m = 10.0
        i = arange(1, self.N + 1)
        # Transform
        #y = x.copy()
        #theta = pi/6
        #for k in range(len(y)-1): # N is odd
        #    if not k%2:
        #        y[k] = x[k]*cos(theta)-x[k+1]*sin(theta)
        #    else:
        #        y[k] = x[k]*cos(theta)+x[k-1]*sin(theta)
        return -sum(sin(x) * sin(i * x ** 2 / pi) ** (2 * m))


class MyPutativeSineEnvelope(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        # 5_dim check
        self.global_optimum = [[-1.906893, -0.796823, 1.906893, 0.796823, -1.906893]]
        self.fglob = -1.49150 * self.N + 1.49150
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        i = arange(0, self.N-1)
        return -sum(
               0.5 + sin(sqrt(x[i] ** 2 + x[i+1] ** 2) - 0.5) ** 2
               / (0.001 * (x[i] ** 2 + x[i+1] ** 2) + 1) ** 2
               )


class MyPutativeEggHolder(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-512.0] * self.N, [512.0] * self.N))
        # 5 dim check
        self.global_optimum = [[485.589834, 436.123707, 451.083199, 466.431218, 421.958519]]
        self.fglob = -915.61991 * self.N + 862.10466
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        i = arange(0, self.N-1)
        return -sum(
               (x[i+1] + 47) * sin(sqrt(abs(x[i+1] + 47 + x[i] / 2)))
               + x[i] * sin(sqrt(abs(x[i] - (x[i+1] + 47))))
               )


class MyPutativeRana(Benchmark):
    def __init__(self, dimensions=5):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-512.0] * self.N, [512.0] * self.N))
        # 5 dim check
        self.global_optimum = [[-512, -512, -512, -512, -511.995602]]
        self.fglob = -511.70430 * self.N + 511.68714
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        i = arange(0, self.N-1)
        return -sum(
               x[i] * cos(sqrt(abs(x[i+1] + x[i] + 1))) * sin(sqrt(abs(x[i+1] - x[i] + 1))) +
               (1 + x[i+1]) * sin(sqrt(abs(x[i+1] + x[i] + 1))) * cos(sqrt(abs(x[i+1] - x[i] + 1)))
               )


class Mytestexp(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-3.0] * self.N, [3.0] * self.N))
        # 5 dim check
        self.global_optimum = [[0,0]]
        self.fglob = -1

    def fun(self, x, *args):
        self.nfev += 1
        x = asarray(x)

        return -(1 - x[0]/4 + x[0]**5 + x[1]**5) * exp(-x[0]**2 - x[1]**2)


if  __name__  == "__main__":
    class_name =  "My" + sys.argv[1]
    check_index = int(sys.argv[2])
    target = globals()[class_name]()
    print("check:", target.fun(target.global_optimum[check_index]))
    print("fglob:", target.fglob)
