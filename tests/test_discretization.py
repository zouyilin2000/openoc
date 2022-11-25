from openoc.discretization import *


def test_xw_lgl():
    x, w = xw_lgl(1)
    assert np.allclose(x, [0])
    assert np.allclose(w, [2])

    x, w = xw_lgl(2)
    assert np.allclose(x, [-1, 1])
    assert np.allclose(w, [1, 1])

    x, w = xw_lgl(3)
    assert np.allclose(x, [-1, 0, 1])
    assert np.allclose(w, [1 / 3, 4 / 3, 1 / 3])

    x, w = xw_lgl(4)
    assert np.allclose(x, [-1, -1 / np.sqrt(5), 1 / np.sqrt(5), 1])
    assert np.allclose(w, [1 / 6, 5 / 6, 5 / 6, 1 / 6])

    x, w = xw_lgl(5)
    assert np.allclose(x, [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1])
    assert np.allclose(w, [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10])

    x, w = xw_lgl(10)
    assert np.allclose(x, [-1, -0.9195339081664588138289, -0.7387738651055050750031, -0.4779249498104444956612,
                           -0.1652789576663870246262, 0.1652789576663870246262, 0.4779249498104444956612,
                           0.7387738651055050750031, 0.9195339081664588138289, 1])
    assert np.allclose(w, [0.02222222222222222222222, 0.1333059908510701111262, 0.2248893420631264521195,
                           0.2920426836796837578756, 0.3275397611838974566565, 0.3275397611838974566565,
                           0.292042683679683757876, 0.224889342063126452119, 0.133305990851070111126,
                           0.02222222222222222222222])


def test_D_lgl():
    x, _ = xw_lgl(10)
    y = x ** 2
    dy = 2 * x
    assert np.allclose(D_lgl(10) @ y, dy)

    x, _ = xw_lgl(10)
    y = x ** 5
    dy = 5 * x ** 4
    assert np.allclose(D_lgl(10) @ y, dy)

    x, _ = xw_lgl(20)
    y = np.sin(x)
    dy = np.cos(x)
    assert np.allclose(D_lgl(20) @ y, dy)


def test_lr_c():
    num_point = np.array([10])
    assert np.allclose(lr_c(num_point)[0], [0])
    assert np.allclose(lr_c(num_point)[1], [10])

    num_point = np.arange(2, 5, dtype=np.int32)
    assert np.allclose(lr_c(num_point)[0], [0, 1, 3])
    assert np.allclose(lr_c(num_point)[1], [2, 4, 7])


def test_lr_nc():
    num_point = np.array([10])
    assert np.allclose(lr_nc(num_point)[0], [0])
    assert np.allclose(lr_nc(num_point)[1], [10])

    num_point = np.arange(2, 5, dtype=np.int32)
    assert np.allclose(lr_nc(num_point)[0], [0, 2, 5])
    assert np.allclose(lr_nc(num_point)[1], [2, 5, 9])


def test_xw_c():
    mesh = np.array([0, 0.1, 0.3, 0.4, 0.7, 0.8, 0.85, 1], dtype=np.float64)
    num_point = np.array([3, 4, 5, 3, 4, 5, 3], dtype=np.int32)
    x, w = xw_c(mesh, num_point)
    y = x ** 2
    assert np.allclose(w @ y, 1 / 3)


def test_xw_nc():
    mesh = np.array([0, 0.1, 0.3, 0.4, 0.7, 0.8, 0.85, 1], dtype=np.float64)
    num_point = np.array([3, 4, 5, 3, 4, 5, 3], dtype=np.int32)
    x, w = xw_nc(mesh, num_point)
    y = x ** 2
    assert np.allclose(w @ y, 1 / 3)


def test_c2nc():
    num_point = np.array([3, 4], dtype=np.int32)
    f_c2nc = c2nc(num_point)
    # 1D
    v_c = np.arange(6, dtype=np.float64)
    assert np.allclose(f_c2nc(v_c), [0, 1, 2, 2, 3, 4, 5])

    # 2D
    v_c = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert np.allclose(f_c2nc(v_c), [[0, 1], [2, 3], [4, 5], [4, 5], [6, 7], [8, 9], [10, 11]])


def test_nc2c():
    num_point = np.array([3, 4], dtype=np.int32)
    f_nc2c = nc2c(num_point)
    # 1D
    v_nc = np.arange(7, dtype=np.float64)
    assert np.allclose(f_nc2c(v_nc), [0, 1, 5, 4, 5, 6])

    # 2D
    v_nc = np.arange(14, dtype=np.float64).reshape(7, 2)
    assert np.allclose(f_nc2c(v_nc), [[0, 1], [2, 3], [10, 12], [8, 9], [10, 11], [12, 13]])


def test_D_c():
    mesh = np.array([0, 0.1, 0.3, 0.4, 0.7, 0.8, 0.85, 1], dtype=np.float64)
    num_point = np.array([6, 7, 5, 6, 7, 5, 6], dtype=np.int32)
    f_c2nc = c2nc(num_point)
    x, _ = xw_c(mesh, num_point)
    x_nc = f_c2nc(x)
    y = x ** 2
    dy = 2 * x_nc
    assert np.allclose(D_c(mesh, num_point) @ y, dy)

    y = np.sin(x)
    dy = np.cos(x_nc)
    assert np.allclose(D_c(mesh, num_point) @ y, dy)


def test_D_nc():
    mesh = np.array([0, 0.1, 0.3, 0.4, 0.7, 0.8, 0.85, 1], dtype=np.float64)
    num_point = np.array([6, 7, 5, 6, 7, 5, 6], dtype=np.int32)
    x, _ = xw_nc(mesh, num_point)
    y = x ** 2
    dy = 2 * x
    assert np.allclose(D_nc(mesh, num_point) @ y, dy)

    y = np.sin(x)
    dy = np.cos(x)
    assert np.allclose(D_nc(mesh, num_point) @ y, dy)


def test_lr_v():
    num_point = np.array([2, 3, 4, 5], dtype=np.int32)
    continuity_xu = np.array([True, False, True, False], dtype=np.bool)
    l_v, r_v = lr_v(num_point, continuity_xu)
    # L_c = 11, L_nc = 14
    assert np.allclose(l_v, [0, 11, 25, 36])
    assert np.allclose(r_v, [11, 25, 36, 50])


def test_lr_m():
    num_point = np.array([2, 3, 4, 5], dtype=np.int32)
    continuity_xu = np.array([True, False, True, False], dtype=np.bool)
    l_m, r_m = lr_m(num_point, continuity_xu)
    # L_c = 11, L_nc = 14
    assert np.allclose(l_m, [0, 14, 28, 42])
    assert np.allclose(r_m, [14, 28, 42, 56])


def test_v2m():
    num_point = np.array([2, 3, 4, 5], dtype=np.int32)
    continuity_xu = np.array([True, False, True, False], dtype=np.bool)
    f_v2m = v2m(num_point, continuity_xu)
    L_v = 50
    v = np.arange(L_v, dtype=np.float64)
    m = f_v2m(v)
    l_v, r_v = lr_v(num_point, continuity_xu)
    l_m, r_m = lr_m(num_point, continuity_xu)
    f_c2nc = c2nc(num_point)
    assert np.allclose(m[l_m[0]:r_m[0]], f_c2nc(v[l_v[0]:r_v[0]]))
    assert np.allclose(m[l_m[1]:r_m[1]], v[l_v[1]:r_v[1]])
    assert np.allclose(m[l_m[2]:r_m[2]], f_c2nc(v[l_v[2]:r_v[2]]))
    assert np.allclose(m[l_m[3]:r_m[3]], v[l_v[3]:r_v[3]])


def test_m2v():
    num_point = np.array([2, 3, 4, 5], dtype=np.int32)
    continuity_xu = np.array([True, False, True, False], dtype=np.bool)
    f_m2v = m2v(num_point, continuity_xu)
    L_m = 56
    m = np.arange(L_m, dtype=np.float64)
    v = f_m2v(m)
    l_v, r_v = lr_v(num_point, continuity_xu)
    l_m, r_m = lr_m(num_point, continuity_xu)
    f_nc2c = nc2c(num_point)
    assert np.allclose(v[l_v[0]:r_v[0]], f_nc2c(m[l_m[0]:r_m[0]]))
    assert np.allclose(v[l_v[1]:r_v[1]], m[l_m[1]:r_m[1]])
    assert np.allclose(v[l_v[2]:r_v[2]], f_nc2c(m[l_m[2]:r_m[2]]))
    assert np.allclose(v[l_v[3]:r_v[3]], m[l_m[3]:r_m[3]])


def test_D_x():
    mesh = np.array([0, 0.1, 0.3, 0.4, 0.7, 0.8, 0.85, 1], dtype=np.float64)
    num_point = np.array([6, 7, 5, 6, 7, 5, 6], dtype=np.int32)
    continuity_x = np.array([True, False, True, False], dtype=np.bool)
    t_c, _ = xw_c(mesh, num_point)
    t_nc, _ = xw_nc(mesh, num_point)
    x = np.concatenate([t_c ** 2, t_nc ** 2, t_c ** 3, t_nc ** 3])
    dx = np.concatenate([2 * t_nc, 2 * t_nc, 3 * t_nc ** 2, 3 * t_nc ** 2])
    assert np.allclose(D_x(mesh, num_point, continuity_x) @ x, dx)
