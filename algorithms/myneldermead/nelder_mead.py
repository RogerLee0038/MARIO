import copy

'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''


def solve(f, x_start,
            step=0.1, no_improve_thr=10e-6,
            no_improv_break=0, max_iter=0, max_eval=10, initials=None,
            alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)

        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    evals = 0
    prev_best = f(x_start)
    best_x = x_start
    best = prev_best
    evals += 1
    if max_eval and evals >= max_eval:
        return [best_x, best]
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        if score < best:
            best_x = x
            best = score
        evals += 1
        if max_eval and evals >= max_eval:
            return [best_x, best]
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv_break and no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if rscore < best:
            best_x = xr
            best = rscore
        evals += 1
        if max_eval and evals >= max_eval:
            return [best_x, best]
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < best:
                best_x = xe
                best = escore
            evals += 1
            if max_eval and evals >= max_eval:
                return [best_x, best]
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < best:
            best_x = xc
            best = cscore
        evals += 1
        if max_eval and evals >= max_eval:
            return [best_x, best]
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            if score < best:
                best_x = redx
                best = score
            evals += 1
            if max_eval and evals >= max_eval:
                return [best_x, best]
            nres.append([redx, score])
        res = nres


if __name__ == "__main__":
    # test
    import math
    import numpy as np

    cnt = 0

    def f(x):
        global cnt
        cnt += 1
        print("cnt", cnt)
        return math.sin(x[0]) * math.cos(x[1]) * (1. / (abs(x[2]) + 1))

    print(solve(f, np.array([0., 0., 0.]), max_eval=20))
