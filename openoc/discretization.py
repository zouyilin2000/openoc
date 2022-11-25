import functools
from typing import Tuple

import numba
import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.special
from numpy.typing import NDArray


@functools.lru_cache
def xw_lgl(num_point: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Legendre-Gauss-Lobatto nodes and weights.
    c.f. https://math.stackexchange.com/questions/2058111/how-to-calculate-nodes-and-weights-of-legendre-gauss-lobatto-rule
    """
    if num_point == 1:
        return np.array([0], dtype=np.float64), np.array([2], dtype=np.float64)
    n = num_point - 1
    poly = scipy.special.legendre(n)
    deriv = np.polyder(poly)
    x = [-1.]
    roots = np.roots(deriv)
    for root in roots:
        x.append(root.real)
    x.append(1.)
    x.sort()
    w = [2. / n / (n + 1)]
    for x_i in x[1:-1]:
        w.append(2. / n / (n + 1) / np.polyval(poly, x_i) ** 2)
    w.append(2. / n / (n + 1))
    return np.array(x, dtype=np.float64), np.array(w, dtype=np.float64)


@functools.lru_cache
def D_lgl(num_point: int) -> NDArray[np.float64]:
    """Compute the derivative matrix of the Legendre-Gauss-Lobatto nodes, i.e. x' = D @ x"""
    x, _ = xw_lgl(num_point)
    D = []
    for i in range(num_point):
        y = np.zeros(num_point)
        y[i] = 1
        poly = scipy.interpolate.lagrange(x, y)
        deriv_poly = np.polyder(poly)
        D.append(np.polyval(deriv_poly, x))
    return np.array(D, dtype=np.float64).T


def lr_c(num_point: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Compute the left and right index of continuous variables"""
    l = np.concatenate([[0], np.cumsum(num_point[:-1] - 1)])
    return l, l + num_point


def lr_nc(num_point: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Compute the left and right index of non-continuous variables"""
    return np.concatenate(([0], np.cumsum(num_point[:-1]))), np.cumsum(num_point)


def xw_c(mesh: NDArray[np.float64], num_point: NDArray[np.int32]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the discretization nodes and weights of continuous variables"""
    l_c, r_c = lr_c(num_point)
    L_c = r_c[-1]  # length
    x = np.zeros(L_c)
    w = np.zeros(L_c)
    width = np.diff(mesh)  # length of each interval
    mid = (mesh[1:] + mesh[:-1]) / 2  # mid-point of each interval
    for l, r, n, d, m in zip(l_c, r_c, num_point, width, mid):
        x_, w_ = xw_lgl(n)
        x[l:r] = x_ * d / 2 + m
        w[l:r] += w_ * d / 2  # += since overlap nodes
    return x, w


def xw_nc(mesh: NDArray[np.float64], num_point: NDArray[np.int32]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the discretization nodes and weights of non-continuous variables"""
    l_c, r_c = lr_nc(num_point)
    L_c = r_c[-1]  # length
    x = np.zeros(L_c)
    w = np.zeros(L_c)
    width = np.diff(mesh)  # length of each interval
    mid = (mesh[1:] + mesh[:-1]) / 2  # mid-point of each interval
    for l, r, n, d, m in zip(l_c, r_c, num_point, width, mid):
        x_, w_ = xw_lgl(n)
        x[l:r] = x_ * d / 2 + m
        w[l:r] = w_ * d / 2
    return x, w


def c2nc(num_point: NDArray[np.int32]):
    """Return a closure that Convert continuous discretization to non-continuous discretization"""
    l_c, r_c = lr_c(num_point)
    l_nc, r_nc = lr_nc(num_point)
    L_nc = r_nc[-1]  # length

    @numba.njit
    def c2nc_(c: NDArray[np.float64]) -> NDArray[np.float64]:
        shape = c.shape
        shape = (L_nc,) + shape[1:]
        nc = np.zeros(shape)
        for l_c_, r_c_, l_nc_, r_nc_ in zip(l_c, r_c, l_nc, r_nc):
            nc[l_nc_:r_nc_] = c[l_c_:r_c_]
        return nc

    return c2nc_


def nc2c(num_point: NDArray[np.int32]):
    """Return a closure that Convert non-continuous discretization to continuous discretization"""
    l_c, r_c = lr_c(num_point)
    l_nc, r_nc = lr_nc(num_point)
    L_c = r_c[-1]  # length

    @numba.njit
    def nc2c_(nc: NDArray[np.float64]) -> NDArray[np.float64]:
        shape = nc.shape
        shape = (L_c,) + shape[1:]
        c = np.zeros(shape)
        for l_c_, r_c_, l_nc_, r_nc_ in zip(l_c, r_c, l_nc, r_nc):
            c[l_c_:r_c_] += nc[l_nc_:r_nc_]  # += since overlap nodes
        return c

    return nc2c_


def D_c(mesh: NDArray[np.float64], num_point: NDArray[np.int32]) -> NDArray[np.float64]:
    """Compute the derivative matrix of continuous variables"""
    data = []
    row = []
    col = []
    l_row, r_row = lr_nc(num_point)
    l_col, r_col = lr_c(num_point)
    L_row = r_row[-1]
    L_col = r_col[-1]
    width = np.diff(mesh)  # length of each interval
    for l_r, l_c, n, d in zip(l_row, l_col, num_point, width):
        D = D_lgl(n) / d * 2
        data.extend(D.flatten())
        row.extend(l_r + np.repeat(np.arange(n), n))
        col.extend(l_c + np.tile(np.arange(n), n))
    D_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
    return D_coo.tocsr()


def D_nc(mesh: NDArray[np.float64], num_point: NDArray[np.int32]) -> NDArray[np.float64]:
    """Compute the derivative matrix of non-continuous variables"""
    data = []
    row = []
    col = []
    l_row, r_row = lr_nc(num_point)
    l_col, r_col = lr_nc(num_point)
    L_row = r_row[-1]
    L_col = r_col[-1]
    width = np.diff(mesh)  # length of each interval
    for l_r, l_c, n, d in zip(l_row, l_col, num_point, width):
        D = D_lgl(n) / d * 2
        data.extend(D.flatten())
        row.extend(l_r + np.repeat(np.arange(n), n))
        col.extend(l_c + np.tile(np.arange(n), n))
    D_coo = scipy.sparse.coo_array((data, (row, col)), shape=(L_row, L_col))
    return D_coo.tocsr()


def lr_v(num_point: NDArray[np.int32], continuity_xu: NDArray[np.bool]) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the left and right index of each variable before c2nc"""
    l_c, r_c = lr_c(num_point)
    l_nc, r_nc = lr_nc(num_point)
    L_c = r_c[-1]
    L_nc = r_nc[-1]
    l = [0]
    r = []
    for c in continuity_xu:
        if c:
            r.append(l[-1] + L_c)
            l.append(r[-1])
        else:
            r.append(l[-1] + L_nc)
            l.append(r[-1])
    return np.array(l[:-1]), np.array(r)


def lr_m(num_point: NDArray[np.int32], continuity_xu: NDArray[np.bool]) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the left and right index of each variable after c2nc (i.e. in middle stage)"""
    L_nc = lr_nc(num_point)[1][-1]
    n_xu = len(continuity_xu)
    return L_nc * np.arange(n_xu, dtype=np.int32), L_nc * (np.arange(n_xu, dtype=np.int32) + 1)


def v2m(num_point: NDArray[np.int32], continuity_xu: NDArray[np.bool]):
    """Return a closure that convert overlapped variables to non-overlap (middle stage) variables"""
    l_v, r_v = lr_v(num_point, continuity_xu)
    l_m, r_m = lr_m(num_point, continuity_xu)
    n_xu = len(continuity_xu)
    f_c2nc = c2nc(num_point)

    @numba.njit
    def v2m_(v: NDArray[np.float64]) -> NDArray[np.float64]:
        shape = v.shape
        shape = (r_m[-1],) + shape[1:]
        y = np.zeros(shape)
        for i in range(n_xu):
            if continuity_xu[i]:
                y[l_m[i]:r_m[i]] = f_c2nc(v[l_v[i]:r_v[i]])
            else:
                y[l_m[i]:r_m[i]] = v[l_v[i]:r_v[i]]
        return y

    return v2m_


def m2v(num_point: NDArray[np.int32], continuity_xu: NDArray[np.bool]):
    """Return a closure that convert middle stage variables to overlapped variables"""
    l_v, r_v = lr_v(num_point, continuity_xu)
    l_m, r_m = lr_m(num_point, continuity_xu)
    n_xu = len(continuity_xu)
    f_nc2c = nc2c(num_point)

    @numba.njit
    def m2v_(m: NDArray[np.float64]) -> NDArray[np.float64]:
        shape = m.shape
        shape = (r_v[-1],) + shape[1:]
        v = np.zeros(shape)
        for i in range(n_xu):
            if continuity_xu[i]:
                v[l_v[i]:r_v[i]] = f_nc2c(m[l_m[i]:r_m[i]])
            else:
                v[l_v[i]:r_v[i]] = m[l_m[i]:r_m[i]]
        return v

    return m2v_


def D_x(mesh: NDArray[np.float64], num_point: NDArray[np.int32], continuity_x: NDArray[np.bool]) \
        -> NDArray[np.float64]:
    """Compute the derivative matrix of all state variables"""
    D_c_ = D_c(mesh, num_point)
    D_nc_ = D_nc(mesh, num_point)
    diag = [D_c_ if c else D_nc_ for c in continuity_x]
    return scipy.sparse.block_diag(diag, format='csr')
