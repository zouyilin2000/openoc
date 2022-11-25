import numpy as np
from numpy.typing import NDArray

from .phase import Phase


class BatchIndexArray:
    def __init__(self, data: NDArray[np.float64], l_ind: NDArray[np.int32], r_ind: NDArray[np.int32]) -> None:
        super().__init__()
        if not len(l_ind) == len(r_ind):
            raise ValueError("l_ind and r_ind must have the same length")
        self._data = data
        self._l_ind = l_ind
        self._r_ind = r_ind
        self._n = len(l_ind)

    def __getitem__(self, i: int) -> NDArray[np.float64]:
        return self._data[self._l_ind[i]:self._r_ind[i]]

    def __setitem__(self, i: int, value: NDArray[np.float64]) -> None:
        self._data[self._l_ind[i]:self._r_ind[i]] = value

    def __len__(self) -> int:
        return self._n


class Variable:
    def __init__(self, phase: Phase, data: NDArray[np.float64]) -> None:
        self._data = data
        self._l_v = phase.l_v
        self._r_v = phase.r_v
        self._t_c = phase.t_c
        self._t_nc = phase.t_nc
        self._n_x = phase.n_x
        self._n = phase.n
        self._array_state = BatchIndexArray(data, self._l_v[:self._n_x], self._r_v[:self._n_x])
        self._array_control = BatchIndexArray(data, self._l_v[self._n_x:], self._r_v[self._n_x:])

    @property
    def x(self) -> BatchIndexArray:
        return self._array_state

    @property
    def u(self) -> BatchIndexArray:
        return self._array_control

    @property
    def t_0(self) -> np.float64:
        return self._data[-2]

    @t_0.setter
    def t_0(self, value: np.float64) -> None:
        self._data[-2] = value

    @property
    def t_f(self) -> np.float64:
        return self._data[-1]

    @t_f.setter
    def t_f(self, value: np.float64) -> None:
        self._data[-1] = value

    @property
    def t_c(self) -> NDArray[np.float64]:
        return self._t_c * (self.t_f - self.t_0) + self.t_0

    @property
    def t_nc(self) -> NDArray[np.float64]:
        return self._t_nc * (self.t_f - self.t_0) + self.t_0

    @property
    def data(self) -> NDArray[np.float64]:
        return self._data


def constant_guess(phase: Phase, value: np.float = 1.) -> Variable:
    if not phase.ok:
        raise ValueError("phase is not fully configured")
    v = Variable(phase, np.full(phase.L, value))
    for i in range(phase.n_x):
        if isinstance(phase.bc_0[i], float):
            v.x[i][0] = phase.bc_0[i]
        elif isinstance(phase.bc_f[i], float):
            v.x[i][-1] = phase.bc_f[i]
    if isinstance(phase.t_0, float):
        v.t_0 = phase.t_0
    if isinstance(phase.t_f, float):
        v.t_f = phase.t_f
    return v


def linear_guess(phase: Phase, default: np.float = 1.) -> Variable:
    if not phase.ok:
        raise ValueError("phase is not fully configured")
    v = Variable(phase, np.full(phase.L, default))
    if isinstance(phase.t_0, float):
        v.t_0 = phase.t_0
    if isinstance(phase.t_f, float):
        v.t_f = phase.t_f

    for i in range(phase.n_x):
        if isinstance(phase.bc_0[i], float) and isinstance(phase.bc_f[i], float):
            if phase.c_x[i]:
                v.x[i] = v.t_c * (phase.bc_f[i] - phase.bc_0[i]) + phase.bc_0[i]
            else:
                v.x[i] = v.t_nc * (phase.bc_f[i] - phase.bc_0[i]) + phase.bc_0[i]
        elif isinstance(phase.bc_0[i], float):
            if phase.c_x[i]:
                v.x[i] = v.t_c * (default - phase.bc_0[i]) + phase.bc_0[i]
            else:
                v.x[i] = v.t_nc * (default - phase.bc_0[i]) + phase.bc_0[i]
        elif isinstance(phase.bc_f[i], float):
            if phase.c_x[i]:
                v.x[i] = v.t_c * (phase.bc_f[i] - default) + default
            else:
                v.x[i] = v.t_nc * (phase.bc_f[i] - default) + default
    return v

