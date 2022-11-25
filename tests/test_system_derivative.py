import numpy as np
import sympy

from openoc.system import System


# here we assume the objective & constraints functions are correct and test all other derivatives

def test_system_gradient():
    s = System(2)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization([0, 0.2, 1], [3, 4], True, False) \
        .set_integral([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0]) + 3 * sympy.cos(p.x[0]) * p.t
                       + 4 * p.u[0] * sympy.cos(s.s[0]) + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[1] * sympy.cos(p.t),
                       6 * sympy.cos(p.x[0]) * p.u[0] + 5 * p.x[0] * sympy.cos(s.s[0]) + 4 * sympy.cos(p.x[0]) * p.t
                       + 3 * p.u[0] * sympy.cos(s.s[0]) + 2 * sympy.cos(p.u[0]) * p.t + s.s[1] * sympy.cos(p.t)])
    s.set_phase([p]).set_objective(sympy.cos(p.I[0]) * s.s[0] * s.s[1] * p.I[1])

    assert p.L == 15
    x = np.arange(17) / 10 + 1
    x[14] += 1

    fd = np.zeros(17)
    eps = 1e-6
    for i in range(17):
        x[i] += eps
        fp = s.objective(x)
        x[i] -= 2 * eps
        fm = s.objective(x)
        x[i] += eps
        fd[i] = (fp - fm) / (2 * eps)

    assert np.allclose(s.gradient(x), fd)


def test_system_jacobian():
    s = System(2)
    p = s.new_phase(1, 1)
    p.set_dynamics([p.x[0] * sympy.cos(s.s[0]) / p.u[0] + p.t ** 2]) \
        .set_boundary_condition([sympy.cos(s.s[1])], [0], 0, None) \
        .set_integral([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0]) + 3 * sympy.cos(p.x[0]) * p.t
                       + 4 * p.u[0] * sympy.cos(s.s[0]) + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[1] * sympy.cos(p.t)]) \
        .set_discretization([0, 0.2, 1], [3, 4], True, False) \
        .set_phase_constraint([p.t - p.x[0] * p.u[0]], [0], [0])
    s.set_phase([p]).set_objective(0).set_system_constraint([s.s[0] ** 2, s.s[1] / 2 * p.I[0]], [0, 0], [0, 0])

    assert p.L == 15
    x = np.arange(17) / 10 + 1
    x[14] += 1

    assert s.constraints(x).shape == (2 + 1 * 7 + 1 * 7 + 1,)  # (17, )

    fd = np.zeros((17, 17))
    eps = 1e-6
    for i in range(17):
        x[i] += eps
        fp = s.constraints(x)
        x[i] -= 2 * eps
        fm = s.constraints(x)
        x[i] += eps
        fd[:, i] = (fp - fm) / (2 * eps)

    sym = np.zeros((17, 17))
    data = s.jacobian(x)
    x, y = s.jacobianstructure()
    for x_, y_, d in zip(x, y, data):
        sym[x_, y_] += d

    assert np.allclose(sym, fd)

def test_system_hessian_objective():
    s = System(2)
    p = s.new_phase(1, 1)
    p.set_dynamics([p.x[0] * sympy.cos(s.s[0]) / p.u[0] + p.t ** 2]) \
        .set_boundary_condition([sympy.cos(s.s[1])], [0], 0, None) \
        .set_integral([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0]) + 3 * sympy.cos(p.x[0]) * p.t
                       + 4 * p.u[0] * sympy.cos(s.s[0]) + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[1] * sympy.cos(p.t)]) \
        .set_discretization([0, 0.2, 1], [3, 4], True, False) \
        .set_phase_constraint([p.t - p.x[0] * p.u[0]], [0], [0])
    s.set_phase([p]).set_objective(0).set_system_constraint([s.s[0] ** 2, s.s[1] / 2 * p.I[0]], [0, 0], [0, 0])

    assert p.L == 15
    x = np.arange(17) / 10 + 1
    x[14] += 1

    assert s.constraints(x).shape == (2 + 1 * 7 + 1 * 7 + 1,)  # (17, )

    fd = np.zeros((17, 17))
    eps = 5e-4
    for i in range(17):
        for j in range(i + 1):
            x[i] += eps
            x[j] += eps
            fpp = s.objective(x)
            x[i] -= 2 * eps
            fmp = s.objective(x)
            x[j] -= 2 * eps
            fmm = s.objective(x)
            x[i] += 2 * eps
            fpm = s.objective(x)
            x[i] -= eps
            x[j] += eps
            fd[i, j] = (fpp - fpm - fmp + fmm) / eps / eps / 4
    sym = np.zeros((17, 17))
    data = s.hessian(x, np.zeros(17), 1)
    x, y = s.hessianstructure()
    for x_, y_, d in zip(x, y, data):
        sym[x_, y_] += d

    assert np.allclose(sym, fd)

def test_system_hessian_constraints():
    s = System(2)
    p = s.new_phase(1, 1)
    p.set_dynamics([p.x[0] * sympy.cos(s.s[0]) / p.u[0] + p.t ** 2]) \
        .set_boundary_condition([sympy.cos(s.s[1])], [0], 0, None) \
        .set_integral([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0]) + 3 * sympy.cos(p.x[0]) * p.t
                       + 4 * p.u[0] * sympy.cos(s.s[0]) + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[1] * sympy.cos(p.t)]) \
        .set_discretization([0, 0.2, 1], [3, 4], True, False) \
        .set_phase_constraint([p.t - p.x[0] * p.u[0]], [0], [0])
    s.set_phase([p]).set_objective(0).set_system_constraint([s.s[0] ** 2, s.s[1] / 2 * p.I[0]], [0, 0], [0, 0])

    assert p.L == 15
    x = np.arange(17, dtype=np.float64) / 10 + 1
    x[13] -= 10
    x[14] += 10

    assert s.constraints(x).shape == (2 + 1 * 7 + 1 * 7 + 1,)  # (17, )

    for c_ in range(17):
        fd = np.zeros((17, 17))
        eps = 5e-4
        for i in range(17):
            for j in range(i + 1):
                x[i] += eps
                x[j] += eps
                fpp = s.constraints(x)[c_]
                x[i] -= 2 * eps
                fmp = s.constraints(x)[c_]
                x[j] -= 2 * eps
                fmm = s.constraints(x)[c_]
                x[i] += 2 * eps
                fpm = s.constraints(x)[c_]
                x[i] -= eps
                x[j] += eps
                fd[i, j] = (fpp - fpm - fmp + fmm) / eps / eps / 4
        sym = np.zeros((17, 17))
        fct_c = np.zeros(17, dtype=np.float64)
        fct_c[c_] = 1.
        data = s.hessian(x, fct_c, 0.)
        h_x, h_y = s.hessianstructure()
        for x_, y_, d in zip(h_x, h_y, data):
            sym[x_, y_] += d

        assert np.allclose(sym, fd, atol=1e-6, rtol=1e-6)