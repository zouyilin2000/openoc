import sympy

from openoc.discretization import *
from openoc.system import System


def test_basic_value_no_phase():
    s = System(2)
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(np.ones(2))
    assert np.allclose(sp, np.ones(2))
    assert np.allclose(V, [])
    assert np.allclose(SP, [])
    assert np.allclose(MT, [])
    assert np.allclose(DT, [])
    assert np.allclose(T, [])
    assert np.allclose(M, [])
    assert np.allclose(I, [])
    assert np.allclose(v_s, np.ones(2))


def test_basic_value_no_static_parameter():
    s = System(0)
    p0 = s.new_phase(1, 1)
    p0.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, 3, True, False).set_integral([p0.x[0] ** 2])
    p1 = s.new_phase(1, 1)
    p1.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1).set_discretization(2, 3, False, True)
    s.set_phase([p0, p1])
    assert p0.L == 13
    assert p1.L == 13
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(np.arange(26))
    assert np.allclose(sp, np.array([]))
    assert np.allclose(V[0], np.arange(13))
    assert np.allclose(V[1], np.arange(13, 26))
    assert np.allclose(SP[0], np.array([]))
    assert np.allclose(SP[1], np.array([]))
    assert np.allclose(MT[0], (11 + 12) / 2)
    assert np.allclose(MT[1], (24 + 25) / 2)
    assert np.allclose(DT[0], 1)
    assert np.allclose(DT[1], 1)
    assert np.allclose(T[0], xw_nc(np.linspace(11, 12, 3, endpoint=True), np.array([3, 3]))[0])
    assert np.allclose(T[1], xw_nc(np.linspace(24, 25, 3, endpoint=True), np.array([3, 3]))[0])
    assert np.allclose(M[0], np.concatenate([p0.f_c2nc(np.arange(5)), np.arange(5, 11), T[0]]))
    assert np.allclose(M[1], np.concatenate([np.arange(13, 19), p1.f_c2nc(np.arange(19, 24)), T[1]]))
    assert np.allclose(I[0], p0.f_c2nc(np.arange(5) ** 2) @ p0.w)
    assert np.allclose(v_s, np.array([I[0]]))


def test_basic_value_normal():
    s = System(2)
    p0 = s.new_phase(1, 1)
    p0.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [2, 3], True, False).set_integral([p0.x[0] ** 2])
    p1 = s.new_phase(1, 1)
    p1.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1).set_discretization(2, [3, 4], False, True)
    s.set_phase([p0, p1])
    assert p0.L == 11
    assert p1.L == 15
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(np.arange(28))
    assert np.allclose(sp, np.array([26, 27]))
    assert np.allclose(V[0], np.arange(11))
    assert np.allclose(V[1], np.arange(11, 26))
    assert np.allclose(SP[0], np.concatenate([np.full(5, 26), np.full(5, 27)]))
    assert np.allclose(SP[1], np.concatenate([np.full(7, 26), np.full(7, 27)]))
    assert np.allclose(MT[0], (9 + 10) / 2)
    assert np.allclose(MT[1], (24 + 25) / 2)
    assert np.allclose(DT[0], 1)
    assert np.allclose(DT[1], 1)
    assert np.allclose(T[0], xw_nc(np.linspace(9, 10, 3, endpoint=True), np.array([2, 3]))[0])
    assert np.allclose(T[1], xw_nc(np.linspace(24, 25, 3, endpoint=True), np.array([3, 4]))[0])
    assert np.allclose(M[0], np.concatenate([p0.f_c2nc(np.arange(4)), np.arange(4, 9), SP[0], T[0]]))
    assert np.allclose(M[1], np.concatenate([np.arange(11, 18), p1.f_c2nc(np.arange(18, 24)), SP[1], T[1]]))
    assert np.allclose(I[0], p0.f_c2nc(np.arange(4) ** 2) @ p0.w)
    assert np.allclose(v_s, np.array([26, 27, I[0]]))


def test_basic_gradient_static_parameter():
    s = System(2)
    p0 = s.new_phase(1, 1)
    p0.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, 3, True, False).set_integral([p0.x[0] ** 2])
    p1 = s.new_phase(1, 1)
    p1.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1).set_discretization(2, 3, False, True)
    s.set_phase([p0, p1])
    assert np.allclose(s._basic_gradient_static_parameter(0)[1][0][1], np.array([26]))
    assert np.allclose(s._basic_gradient_static_parameter(1)[1][0][1], np.array([27]))


def test_basic_gradient_integral():
    # basic strategy is to use Finite Difference Method
    # construct an integration with continuous state, non-continuous state, static parameter, and t
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [3, 4], True, False) \
        .set_integral([p.x[0] ** 2 * p.u[0] - p.u[0] * sympy.cos(s.s[0]) * sympy.sqrt(p.t)])
    s.set_phase([p]).set_objective(0)

    def I_func(x):
        sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
        return I[0]

    assert p.L == 15
    x = np.arange(16) / 10 + 1
    x[14] += 1

    fd = np.zeros(16)
    eps = 1e-7
    for i in range(16):
        x[i] += eps
        fp = I_func(x)
        x[i] -= 2 * eps
        fm = I_func(x)
        x[i] += eps
        fd[i] = (fp - fm) / eps / 2

    init, code = s._basic_gradient_integral(0, 0)
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
    exec(init.replace('self.', 's.'), globals(), locals())
    sym = np.zeros(16)
    for code, index in code:
        sym[index] += eval(code)

    assert np.allclose(fd, sym)


def test_basic_gradient_phase_func():
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [3, 4], True, False) \
        .set_phase_constraint([p.x[0] ** 2 * p.u[0] - p.u[0] * sympy.cos(s.s[0]) * sympy.sqrt(p.t)], [0], [1])
    s.set_phase([p]).set_objective(0)

    assert p.l0_m == 7

    def F_func(x):
        sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
        return p.F_c[0].F(M[0], p.l0_m)

    assert p.L == 15
    x = np.arange(16) / 2 + 1
    x[14] += 1

    fd = np.zeros((7, 16))
    eps = 1e-7
    for i in range(16):
        x[i] += eps
        fp = F_func(x)
        x[i] -= 2 * eps
        fm = F_func(x)
        x[i] += eps
        fd[:, i] = (fp - fm) / eps / 2

    init, code = s._basic_gradient_phase_func(0, 'F_c[0]')
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
    exec(init.replace('self.', 's.'), globals(), locals())
    sym = np.zeros((7, 16))
    for code, index in code:
        v = eval(code)
        for i in range(7):
            sym[i, index[i]] += v[i]

    assert np.allclose(fd, sym)


def test_basic_gradient_system():
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [3, 4], True, False) \
        .set_integral([p.x[0] ** 2 * p.u[0] - p.u[0] * sympy.cos(s.s[0]) * sympy.sqrt(p.t)])
    s.set_phase([p]).set_objective(sympy.sin(p.I[0]) * sympy.cos(s.s[0]))

    def O_func(x):
        sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
        return s.F_o.F(v_s, 1)[0]

    assert p.L == 15
    x = np.arange(16) / 10 + 1
    x[14] += 1

    fd = np.zeros(16)
    eps = 1e-7
    for i in range(16):
        x[i] += eps
        fp = O_func(x)
        x[i] -= 2 * eps
        fm = O_func(x)
        x[i] += eps
        fd[i] = (fp - fm) / eps / 2

    init, code = s._basic_gradient_system('F_o')
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
    exec(init.replace('self.', 's.'), globals(), locals())
    sym = np.zeros(16)
    for code, index in code:
        sym[index] += eval(code)

    assert np.allclose(fd, sym)


def test_basic_hessian_integral():
    # the integral has non-zero partial derivative with respect to any two variables
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [3, 4], True, False) \
        .set_integral([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0]) + 3 * sympy.cos(p.x[0]) * p.t
                       + 4 * p.u[0] * sympy.cos(s.s[0]) + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[0] * sympy.cos(p.t)])
    s.set_phase([p]).set_objective(0)

    def I_func(x):
        sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
        return I[0]

    assert p.L == 15
    x = np.arange(16) / 10 + 1
    # x[14] += 1

    fd = np.zeros((16, 16))
    eps = 5e-4
    for i in range(16):
        for j in range(i + 1):
            x[i] += eps
            x[j] += eps
            fpp = I_func(x)
            x[i] -= 2 * eps
            fmp = I_func(x)
            x[j] -= 2 * eps
            fmm = I_func(x)
            x[i] += 2 * eps
            fpm = I_func(x)
            x[i] -= eps
            x[j] += eps
            fd[i, j] = (fpp - fmp - fpm + fmm) / eps / eps / 4

    init, code = s._basic_hessian_integral(0, 0)
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
    exec(init.replace('self.', 's.'), globals(), locals())
    sym = np.zeros((16, 16))
    for c, x_, y_ in code:
        v = eval(c)
        for i in range(len(x_)):
            sym[x_[i], y_[i]] += v[i]

    assert np.allclose(fd, sym)


def test_basic_hessian_phase_func():
    # the phase constraint has non-zero partial derivative with respect to any two variables
    s = System(1)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [3, 4], True, False) \
        .set_phase_constraint([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0])
                               + 3 * sympy.cos(p.x[0]) * p.t + 4 * p.u[0] * sympy.cos(s.s[0])
                               + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[0] * sympy.cos(p.t)],
                              [0], [0])
    s.set_phase([p]).set_objective(0)

    def F_func(x):
        sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
        return p.F_c[0].F(M[0], p.l0_m)

    assert p.L == 15
    x = np.arange(16) / 10 + 1
    x[14] += 1

    fd = np.zeros((7, 16, 16))
    eps = 5e-4
    for i in range(16):
        for j in range(i + 1):
            x[i] += eps
            x[j] += eps
            fpp = F_func(x)
            x[i] -= 2 * eps
            fmp = F_func(x)
            x[j] -= 2 * eps
            fmm = F_func(x)
            x[i] += 2 * eps
            fpm = F_func(x)
            x[i] -= eps
            x[j] += eps
            fd[:, i, j] = (fpp - fmp - fpm + fmm) / eps / eps / 4

    init, code = s._basic_hessian_phase_func(0, 'F_c[0]')
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
    exec(init.replace('self.', 's.'), globals(), locals())
    sym = np.zeros((7, 16, 16))
    for c, x_, y_ in code:
        v = eval(c)
        for i in range(7):
            sym[i, x_[i], y_[i]] += v[i]

    assert np.allclose(fd, sym)


def test_basic_hessian_system():
    s = System(2)
    p = s.new_phase(1, 1)
    p.set_dynamics([0]).set_boundary_condition([0], [0], 0, 1) \
        .set_discretization(2, [3, 4], True, False) \
        .set_integral([sympy.cos(p.x[0]) * p.u[0] + 2 * p.x[0] * sympy.cos(s.s[0]) + 3 * sympy.cos(p.x[0]) * p.t
                       + 4 * p.u[0] * sympy.cos(s.s[0]) + 5 * sympy.cos(p.u[0]) * p.t + 6 * s.s[1] * sympy.cos(p.t),
                       6 * sympy.cos(p.x[0]) * p.u[0] + 5 * p.x[0] * sympy.cos(s.s[0]) + 4 * sympy.cos(p.x[0]) * p.t
                       + 3 * p.u[0] * sympy.cos(s.s[0]) + 2 * sympy.cos(p.u[0]) * p.t + s.s[1] * sympy.cos(p.t)])
    s.set_phase([p]).set_objective(sympy.cos(p.I[0]) * s.s[0] * s.s[1] * p.I[1])

    def O_func(x):
        sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
        return s.F_o.F(v_s, 1)[0]

    assert p.L == 15
    x = np.arange(17) / 10 + 1
    x[14] += 1

    fd = np.zeros((17, 17))
    eps = 1e-4
    for i in range(17):
        for j in range(i + 1):
            x[i] += eps
            x[j] += eps
            fpp = O_func(x)
            x[i] -= 2 * eps
            fmp = O_func(x)
            x[j] -= 2 * eps
            fmm = O_func(x)
            x[i] += 2 * eps
            fpm = O_func(x)
            x[i] -= eps
            x[j] += eps
            fd[i, j] = (fpp - fmp - fpm + fmm) / eps / eps / 4

    init, code = s._basic_hessian_system('F_o')
    sp, V, SP, MT, DT, T, M, I, v_s = s._basic_value(x)
    exec(init.replace('self.', 's.'), globals(), locals())
    sym = np.zeros((17, 17))
    for c, x_, y_ in code:
        v = eval(c)
        for i in range(len(x_)):
            sym[x_[i], y_[i]] += v[i]

    # for i in range(17):
    #     for j in range(17):
    #         if abs(fd[i, j] - sym[i, j]) > 1e-4:
    #             print(i, j, fd[i, j], sym[i, j])

    assert np.allclose(fd, sym)
