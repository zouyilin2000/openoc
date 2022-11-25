from typing import Union, List, Iterable

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from .fastfunc import FastFunc
from .phase import Phase


class System:
    def __init__(self, static_parameter: Union[int, List[str]], /, simplify: bool = False, parallel: bool = False,
                 fastmath: bool = False):
        """Initialize a system with given static parameters.
        If static_parameter is an integer, the static parameters will be named as s_0, ..., s_{n - 1}.

        If simplify is True, every symbolic expression will be simplified (by sympy.simplify) before being compiled.
        This may slow down the speed of compilation.

        If parallel is True, parallel flag will be passed to numba JIT compiler,
        which will generate parallel code for multicore CPUs.
        This may slow down the speed of compilation.

        If fastmath is True, fastmath flag will be passed to numba JIT compiler,
        see numba & LLVM documentations for details.

        Args:
            static_parameter (Union[int, List[str]]): Number of static parameters or list of static parameter names.
            simplify (bool, optional): Whether to use sympy to simplify Exprs. Defaults to False.
            parallel (bool, optional): Whether to use numba parallel mode. Defaults to False.
            fastmath (bool, optional): Whether to use numba fastmath mode. Defaults to False.
        """
        if isinstance(static_parameter, int):
            self._num_static_parameter = static_parameter
            self._name_static_parameter = [f's_{i}' for i in range(static_parameter)]
        elif isinstance(static_parameter, list):
            self._name_static_parameter = static_parameter
            self._num_static_parameter = len(static_parameter)
        else:
            raise ValueError('static_parameter must be int or list of str')

        self._symbol_static_parameter = [sp.Symbol(name) for name in self._name_static_parameter]
        self._symbols = self._symbol_static_parameter

        self._identifier_phase = 0

        self._simplify = simplify
        self._parallel = parallel
        self._fastmath = fastmath

        self._phase_set = False
        self._objective_set = False  # user must set an objective function
        self._system_constraint_set = False
        self.set_phase([])  # no phase is a valid system and is set by default
        self.set_system_constraint([], np.array([]), np.array([]))  # no system constraint by default

    def new_phase(self, state: Union[int, List[str]], control: Union[int, List[str]]):
        """Create a new phase for the given system.

        Args:
            state (Union[int, List[str]]): Number of state variables or list of state variable names.
            control (Union[int, List[str]]): Number of control variables or list of control variable names.

        Returns:
            Phase: A new phase, with static_parameter and configs correctly set.
        """
        self._identifier_phase += 1
        return Phase(state, control, self._symbol_static_parameter, self._identifier_phase - 1, simplify=self._simplify,
                     parallel=self._parallel, fastmath=self._fastmath)

    def set_phase(self, phase: List[Phase]):
        """Set the phases of the system.

        Args:
            phase (Phase): Phases of the system.
        """
        for i, p in enumerate(phase):
            if not p.ok:
                raise ValueError(f'Dynamics, boundary conditions, '
                                 f'or discretization scheme of phase {i} are not fully set')
        self._phase = phase
        self._num_phase = len(phase)

        self._phase_set = True

        self._update_lr_phase()
        self._update_symbols()
        if self._objective_set and self._system_constraint_set:
            self._update_jacobian_structure()
            self._update_hessian_structure()
        if self._system_constraint_set:
            self._update_bounds()
        return self

    def set_objective(self, objective: Union[float, sp.Expr]):
        """Set the objective of the system.

        Args:
            objective (Union[int, float, sp.Expr]): Objective of the system composed by s & I.
        """
        self._expr_objective = sp.sympify(objective)
        self._func_objective = FastFunc(self._expr_objective, self._symbols, self._simplify, self._parallel,
                                        self._fastmath)
        self._objective_set = True

        if self._phase_set and self._system_constraint_set:
            self._update_jacobian_structure()
            self._update_hessian_structure()
        return self

    def set_system_constraint(self, constraint: List[sp.Expr], lower_bound: Iterable[float],
                              upper_bound: Iterable[float]):
        """Set the system constraint of the system.

        Args:
            constraint (List[Union[int, float, sp.Expr]]): Global constraint of the system composed by s & I.
            lower_bound (NDArray[np.float64]): Lower bound of the constraints.
            upper_bound (NDArray[np.float64]): Upper bound of the constraints.
        """
        lower_bound = list(lower_bound)
        upper_bound = list(upper_bound)
        if not len(constraint) == len(lower_bound) == len(upper_bound):
            raise ValueError('constraint, lower_bound and upper_bound must have the same length')

        self._static_parameter_bounds_system = []
        self._expr_system_constraint = []
        lower_bound_system_constraint = []
        upper_bound_system_constraint = []
        for c, lb, ub in zip(constraint, lower_bound, upper_bound):
            if c.is_symbol and c in self.s:
                self._static_parameter_bounds_system.append((self.s.index(c), lb, ub))
            else:
                self._expr_system_constraint.append(sp.sympify(c))
                lower_bound_system_constraint.append(lb)
                upper_bound_system_constraint.append(ub)

        self._func_system_constraint = [FastFunc(c, self._symbols, self._simplify, self._parallel, self._fastmath)
                                        for c in self._expr_system_constraint]
        self._num_system_constraint = len(self._expr_system_constraint)
        self._lower_bound_system_constraint = np.array(lower_bound_system_constraint)
        self._upper_bound_system_constraint = np.array(upper_bound_system_constraint)

        self._system_constraint_set = True

        if self._phase_set:
            self._update_bounds()
        if self._objective_set and self._phase_set:
            self._update_jacobian_structure()
            self._update_hessian_structure()
        return self

    def _update_lr_phase(self):
        if self.n_p == 0:
            self.l_p = np.array([], dtype=np.int32)
            self.r_p = np.array([], dtype=np.int32)
            self.l_s = 0
            self.r_s = self.n_s
        else:
            l_p = [0]
            r_p = []
            for p in self.p:
                r_p.append(l_p[-1] + p.L)
                l_p.append(r_p[-1])
            self.l_p = np.array(l_p[:-1], dtype=np.int32)
            self.r_p = np.array(r_p, dtype=np.int32)
            self.l_s = r_p[-1]
            self.r_s = self.l_s + self.n_s

    def _update_symbols(self):
        self._symbols = self._symbol_static_parameter
        mapping_integral = []
        for i, p in enumerate(self.p):
            self._symbols += p.I
            mapping_integral.extend([(i, j) for j in range(p.n_I)])
        self._num_symbol = len(self._symbols)
        self._mapping_integral = mapping_integral

    def _update_bounds(self):
        lower_bound_static_parameter = np.full(self.n_s, -np.inf, dtype=np.float64)
        upper_bound_static_parameter = np.full(self.n_s, np.inf, dtype=np.float64)
        for p in self.p:
            for i, lb, ub in p.s_b:
                lower_bound_static_parameter[i] = np.maximum(lower_bound_static_parameter[i], lb)
                upper_bound_static_parameter[i] = np.minimum(upper_bound_static_parameter[i], ub)
        for i, lb, ub in self._static_parameter_bounds_system:
            lower_bound_static_parameter[i] = np.maximum(lower_bound_static_parameter[i], lb)
            upper_bound_static_parameter[i] = np.minimum(upper_bound_static_parameter[i], ub)

        self._lower_bound_variable = \
            np.concatenate([p.v_lb for p in self.p] + [np.array(lower_bound_static_parameter)])
        self._upper_bound_variable = \
            np.concatenate([p.v_ub for p in self.p] + [np.array(upper_bound_static_parameter)])

        lower_bound_constraint = []
        upper_bound_constraint = []
        # system constraint
        lower_bound_constraint.append(self._lower_bound_system_constraint)
        upper_bound_constraint.append(self._upper_bound_system_constraint)
        # dynamics
        lower_bound_constraint.extend(np.zeros(p.n_x * p.l0_m, dtype=np.float64) for p in self.p)
        upper_bound_constraint.extend(np.zeros(p.n_x * p.l0_m, dtype=np.float64) for p in self.p)
        # phase constraint
        lower_bound_constraint.extend(np.repeat(p.c_lb, p.l0_m) for p in self.p)
        upper_bound_constraint.extend(np.repeat(p.c_ub, p.l0_m) for p in self.p)
        # boundary condition
        lower_bound_constraint.extend(np.zeros(p.n_bc, dtype=np.float64) for p in self.p)
        upper_bound_constraint.extend(np.zeros(p.n_bc, dtype=np.float64) for p in self.p)

        self._lower_bound_constraint = np.concatenate(lower_bound_constraint)
        self._upper_bound_constraint = np.concatenate(upper_bound_constraint)

    @property
    def n_s(self) -> int:
        """Number of static parameters."""
        return self._num_static_parameter

    @property
    def s(self) -> List[sp.Symbol]:
        """Sympy symbols of static parameters."""
        return self._symbol_static_parameter

    @property
    def n_p(self) -> int:
        """Number of phases."""
        return self._num_phase

    @property
    def p(self) -> List[Phase]:
        """Phases of system."""
        return self._phase

    @property
    def F_o(self) -> FastFunc:
        """FastFuncs of objective function."""
        return self._func_objective

    @property
    def n_c(self) -> int:
        """Number of system constraints."""
        return self._num_system_constraint

    @property
    def F_c(self) -> List[FastFunc]:
        """FastFuncs of system constraints."""
        return self._func_system_constraint

    @property
    def v_lb(self) -> NDArray[np.float64]:
        """Lower bound of variables."""
        return self._lower_bound_variable

    @property
    def v_ub(self) -> NDArray[np.float64]:
        """Upper bound of variables."""
        return self._upper_bound_variable

    @property
    def c_lb(self) -> NDArray[np.float64]:
        """Lower bound of constraints."""
        return self._lower_bound_constraint

    @property
    def c_ub(self) -> NDArray[np.float64]:
        """Upper bound of constraints."""
        return self._upper_bound_constraint

    @property
    def N(self) -> int:
        """Number of phases."""
        return self.n_p

    @property
    def L(self) -> int:
        return self.r_s

    @property
    def ok(self) -> bool:
        return self._objective_set

    def _basic_value(self, x):
        sp = x[self.l_s:self.r_s]
        V = [x[l:r] for l, r in zip(self.l_p, self.r_p)]
        SP = [np.repeat(sp, p.l0_m) for p in self.p]
        MT = [(v[-1] + v[-2]) / 2 for v in V]
        DT = [v[-1] - v[-2] for v in V]
        T = [(p.t_m - 0.5) * dt + mt for p, v, mt, dt in zip(self.p, V, MT, DT)]
        M = [np.concatenate((p.f_v2m(v[:-2]), s, t)) for p, v, s, t in zip(self.p, V, SP, T)]
        I = [f_I.F(m, p.l0_m) @ p.w * dt for p, m, dt in zip(self.p, M, DT) for f_I in p.F_I]
        v_s = np.concatenate([sp, I])
        return sp, V, SP, MT, DT, T, M, I, v_s

    def _basic_gradient_static_parameter(self, sp_, suffix=''):
        return '', [('np.array([1.], dtype=np.float64)', np.array([self.l_s + sp_], dtype=np.int32))]

    def _basic_gradient_integral(self, p_, I_, suffix=''):
        p = self.p[p_]
        init = ''
        grad = []
        init += f'p_gi{suffix} = self.p[{p_}]\n'
        init += f'f_gi{suffix} = p_gi{suffix}.F_I[{I_}].F(M[{p_}], p_gi{suffix}.l0_m) @ p_gi{suffix}.w\n'
        init += f'g_gi{suffix} = p_gi{suffix}.F_I[{I_}].G(M[{p_}], p_gi{suffix}.l0_m) * p_gi{suffix}.w * DT[{p_}]\n'
        for row, n_ in enumerate(p.F_I[I_].G_index):
            if n_ < p.n:
                if p.c[n_]:
                    grad.append(
                        (f'p_gi{suffix}.f_nc2c(g_gi{suffix}[{row}])', np.arange(p.l_v[n_], p.r_v[n_]) + self.l_p[p_]))
                else:
                    grad.append((f'g_gi{suffix}[{row}]', np.arange(p.l_v[n_], p.r_v[n_]) + self.l_p[p_]))
            elif n_ < p.n + self.n_s:
                grad.append(
                    (f'np.array([np.sum(g_gi{suffix}[{row}])], dtype=np.float64)', np.array([self.l_s + n_ - p.n])))
            else:
                grad.append((f'np.array([g_gi{suffix}[{row}] @ (1 - p_gi{suffix}.t_m),'
                             f' g_gi{suffix}[{row}] @ p_gi{suffix}.t_m])',
                             np.array([self.r_p[p_] - 2, self.r_p[p_] - 1])))
        grad.append((f'np.array([-f_gi{suffix}, f_gi{suffix}], dtype=np.float64)',
                     np.array([self.r_p[p_] - 2, self.r_p[p_] - 1])))
        return init, grad

    def _basic_gradient_phase_func(self, p_: int, func_name: str, suffix=''):
        p = self.p[p_]
        init = ''
        grad = []
        init += f'p_gpf{suffix} = self.p[{p_}]\n'
        init += f'g_gpf{suffix} = p_gpf{suffix}.{func_name}.G(M[{p_}], p_gpf{suffix}.l0_m)\n'
        func = eval(f'p.{func_name}')
        for row, n_ in enumerate(func.G_index):
            if n_ < p.n:
                if p.c[n_]:
                    v_index = np.concatenate(
                        [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[n_] + self.l_p[p_] for i in range(p.N)])
                    grad.append((f'g_gpf{suffix}[{row}]', v_index))
                else:
                    grad.append((f'g_gpf{suffix}[{row}]', np.arange(p.l_v[n_], p.r_v[n_]) + self.l_p[p_]))
            elif n_ < p.n + self.n_s:
                grad.append((f'g_gpf{suffix}[{row}]',
                             np.full(p.l0_m, self.l_s + n_ - p.n)))
            else:
                grad.append(
                    (f'(g_gpf{suffix}[{row}] * (1 - p_gpf{suffix}.t_m))', np.full(p.l0_m, self.r_p[p_] - 2)))
                grad.append((f'(g_gpf{suffix}[{row}] * p_gpf{suffix}.t_m)', np.full(p.l0_m, self.r_p[p_] - 1)))
        return init, grad

    def _basic_gradient_system(self, func_name: str, suffix=''):
        init = f'g_gs{suffix} = self.{func_name}.G(v_s, 1)\n'
        grad = []
        func = eval('self.' + func_name)
        for row, n_ in enumerate(func.G_index):
            if n_ < self.n_s:
                init_, grad_ = self._basic_gradient_static_parameter(n_, suffix=f'_gs_sp{n_}{suffix}')
                init += init_
                grad.extend(map(lambda c, i: (c + f' * g_gs{suffix}[{row}]', i), *zip(*grad_)))
            else:
                p_, I_ = self._mapping_integral[n_ - self.n_s]
                init_, grad_ = self._basic_gradient_integral(p_, I_, suffix=f'_gs_i{p_}{I_}{suffix}')
                init += init_
                grad.extend(map(lambda c, i: (c + f' * g_gs{suffix}[{row}]', i), *zip(*grad_)))
        return init, grad

    def _basic_hessian_integral(self, p_, I_, suffix=''):
        p = self.p[p_]
        init = ''
        hess = []  # code, row, col (row >= col)
        init += f'p_hi{suffix} = self.p[{p_}]\n'
        init += f'g_hi{suffix} = p_hi{suffix}.F_I[{I_}].G(M[{p_}], p_hi{suffix}.l0_m) * p_hi{suffix}.w\n'
        init += f'h_hi{suffix} = p_hi{suffix}.F_I[{I_}].H(M[{p_}], p_hi{suffix}.l0_m) * p_hi{suffix}.w * DT[{p_}]\n'
        # 2 parts: v/v, dt/v (dt/dt = 0)
        # v/v
        for row, (x_, y_) in enumerate(zip(*p.F_I[I_].H_index)):  # x_ >= y_
            if x_ < p.n:
                if p.c[x_]:
                    x_index = np.concatenate(
                        [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[x_] + self.l_p[p_] for i in range(p.N)])
                else:
                    x_index = np.arange(p.l_v[x_], p.r_v[x_]) + self.l_p[p_]
                if p.c[y_]:
                    y_index = np.concatenate(
                        [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                else:
                    y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                hess.append((f'h_hi{suffix}[{row}]', x_index, y_index))
            elif x_ < p.n + self.n_s:
                if y_ < p.n:
                    if p.c[y_]:
                        y_index = np.concatenate(
                            [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                    else:
                        y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                    hess.append((f'h_hi{suffix}[{row}]', np.full(p.l0_m, self.l_s + x_ - p.n), y_index))
                else:
                    hess.append((f'np.array([np.sum(h_hi{suffix}[{row}])], dtype=np.float64)',
                                 np.array([self.l_s + x_ - p.n]), np.array([self.l_s + y_ - p.n])))
            else:
                if y_ < p.n:
                    if p.c[y_]:
                        y_index = np.concatenate(
                            [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                    else:
                        y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                    hess.append(
                        (f'h_hi{suffix}[{row}] * (1 - p_hi{suffix}.t_m)', np.full(p.l0_m, self.r_p[p_] - 2), y_index))
                    hess.append((f'h_hi{suffix}[{row}] * p_hi{suffix}.t_m', np.full(p.l0_m, self.r_p[p_] - 1), y_index))
                elif y_ < p.n + self.n_s:
                    hess.append((f'np.array([np.sum(h_hi{suffix}[{row}] * (1 - p_hi{suffix}.t_m))], dtype=np.float64)',
                                 np.array([self.l_s + y_ - p.n]), np.array([self.r_p[p_] - 2])))
                    hess.append((f'np.array([np.sum(h_hi{suffix}[{row}] * p_hi{suffix}.t_m)], dtype=np.float64)',
                                 np.array([self.l_s + y_ - p.n]), np.array([self.r_p[p_] - 1])))
                else:
                    hess.append((
                        f'np.array([np.sum(h_hi{suffix}[{row}] * (1 - p_hi{suffix}.t_m) * (1 - p_hi{suffix}.t_m))], dtype=np.float64)',
                        np.array([self.r_p[p_] - 2]), np.array([self.r_p[p_] - 2])))
                    hess.append((
                        f'np.array([np.sum(h_hi{suffix}[{row}] * p_hi{suffix}.t_m * (1 - p_hi{suffix}.t_m))], dtype=np.float64)',
                        np.array([self.r_p[p_] - 1]), np.array([self.r_p[p_] - 2])))
                    hess.append((
                        f'np.array([np.sum(h_hi{suffix}[{row}] * p_hi{suffix}.t_m * p_hi{suffix}.t_m)], dtype=np.float64)',
                        np.array([self.r_p[p_] - 1]), np.array([self.r_p[p_] - 1])))
        # t/v
        for row, y_ in enumerate(p.F_I[I_].G_index):  # x_ >= y_
            if y_ < p.n:
                if p.c[y_]:
                    y_index = np.concatenate(
                        [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                else:
                    y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                hess.append((f'-g_hi{suffix}[{row}]', np.full(p.l0_m, self.r_p[p_] - 2), y_index))
                hess.append((f'g_hi{suffix}[{row}]', np.full(p.l0_m, self.r_p[p_] - 1), y_index))
            elif y_ < p.n + self.n_s:
                hess.append(
                    (f'np.array([-np.sum(g_hi{suffix}[{row}])], dtype=np.float64)', np.array([self.l_s + y_ - p.n]),
                     np.array([self.r_p[p_] - 2])))
                hess.append(
                    (f'np.array([np.sum(g_hi{suffix}[{row}])], dtype=np.float64)', np.array([self.l_s + y_ - p.n]),
                     np.array([self.r_p[p_] - 1])))
            else:
                hess.append((f'np.array([2 * -np.sum(g_hi{suffix}[{row}] * (1 - p_hi{suffix}.t_m))], dtype=np.float64)',
                             np.array([self.r_p[p_] - 2]), np.array([self.r_p[p_] - 2])))
                hess.append((f'np.array([np.sum(g_hi{suffix}[{row}] * (1 - p_hi{suffix}.t_m * 2))], dtype=np.float64)',
                             np.array([self.r_p[p_] - 1]), np.array([self.r_p[p_] - 2])))
                hess.append((f'np.array([2 * np.sum(g_hi{suffix}[{row}] * p_hi{suffix}.t_m)], dtype=np.float64)',
                             np.array([self.r_p[p_] - 1]), np.array([self.r_p[p_] - 1])))
        return init, hess

    def _basic_hessian_phase_func(self, p_: int, func_name: str, suffix=''):
        p = self.p[p_]
        init = ''
        hess = []  # code, row, col (row >= col)
        init += f'p_hpf{suffix} = self.p[{p_}]\n'
        init += f'g_hpf{suffix} = p_hpf{suffix}.{func_name}.G(M[{p_}], p_hpf{suffix}.l0_m)\n'
        init += f'h_hpf{suffix} = p_hpf{suffix}.{func_name}.H(M[{p_}], p_hpf{suffix}.l0_m)\n'
        func = eval('p.' + func_name)
        for row, (x_, y_) in enumerate(zip(*func.H_index)):  # x_ >= y_
            if x_ < p.n:
                if p.c[x_]:
                    x_index = np.concatenate(
                        [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[x_] + self.l_p[p_] for i in range(p.N)])
                else:
                    x_index = np.arange(p.l_v[x_], p.r_v[x_]) + self.l_p[p_]
                if p.c[y_]:
                    y_index = np.concatenate(
                        [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                else:
                    y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                hess.append((f'h_hpf{suffix}[{row}]', x_index, y_index))
            elif x_ < p.n + self.n_s:
                if y_ < p.n:
                    if p.c[y_]:
                        y_index = np.concatenate(
                            [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                    else:
                        y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                    hess.append((f'h_hpf{suffix}[{row}]', np.full(p.l0_m, self.l_s + x_ - p.n), y_index))
                else:
                    hess.append((f'h_hpf{suffix}[{row}]', np.full(p.l0_m, self.l_s + x_ - p.n),
                                 np.full(p.l0_m, self.l_s + y_ - p.n)))
            else:
                if y_ < p.n:
                    if p.c[y_]:
                        y_index = np.concatenate(
                            [np.arange(p.l_c[i], p.r_c[i]) + p.l_v[y_] + self.l_p[p_] for i in range(p.N)])
                    else:
                        y_index = np.arange(p.l_v[y_], p.r_v[y_]) + self.l_p[p_]
                    hess.append(
                        (f'h_hpf{suffix}[{row}] * (1 - p_hpf{suffix}.t_m)', np.full(p.l0_m, self.r_p[p_] - 2), y_index))
                    hess.append(
                        (f'h_hpf{suffix}[{row}] * p_hpf{suffix}.t_m', np.full(p.l0_m, self.r_p[p_] - 1), y_index))
                elif y_ < p.n + self.n_s:
                    hess.append((f'h_hpf{suffix}[{row}] * (1 - p_hpf{suffix}.t_m)',
                                 np.full(p.l0_m, self.l_s + y_ - p.n), np.full(p.l0_m, self.r_p[p_] - 2)))
                    hess.append((f'h_hpf{suffix}[{row}] * p_hpf{suffix}.t_m',
                                 np.full(p.l0_m, self.l_s + y_ - p.n), np.full(p.l0_m, self.r_p[p_] - 1)))
                else:
                    hess.append((f'h_hpf{suffix}[{row}] * (1 - p_hpf{suffix}.t_m) * (1 - p_hpf{suffix}.t_m)',
                                 np.full(p.l0_m, self.r_p[p_] - 2), np.full(p.l0_m, self.r_p[p_] - 2)))
                    hess.append((f'h_hpf{suffix}[{row}] * p_hpf{suffix}.t_m * (1 - p_hpf{suffix}.t_m)',
                                 np.full(p.l0_m, self.r_p[p_] - 1), np.full(p.l0_m, self.r_p[p_] - 2)))
                    hess.append((f'h_hpf{suffix}[{row}] * p_hpf{suffix}.t_m * p_hpf{suffix}.t_m',
                                 np.full(p.l0_m, self.r_p[p_] - 1), np.full(p.l0_m, self.r_p[p_] - 1)))
        return init, hess

    def _basic_hessian_system(self, func_name: str, suffix=''):
        init = (f'g_hs{suffix} = self.{func_name}.G(v_s, 1)\n'
                f'h_hs{suffix} = self.{func_name}.H(v_s, 1)\n')
        hess = []
        func = eval('self.' + func_name)
        for row, n_ in enumerate(func.G_index):
            if n_ >= self.n_s:
                p_, I_ = self._mapping_integral[n_ - self.n_s]
                init_, hess_ = self._basic_hessian_integral(p_, I_, suffix=f'_hs_i{p_}{I_}{suffix}')
                init += init_
                for code, x_, y_ in hess_:
                    hess.append((code + f' * g_hs{suffix}[{row}]', x_, y_))

        for row, (x_, y_) in enumerate(zip(*func.H_index)):
            if x_ < self.n_s:
                init_x, grad_x = self._basic_gradient_static_parameter(x_, suffix=f'_hs_sp{y_}{suffix}')
            else:
                p_, I_ = self._mapping_integral[x_ - self.n_s]
                init_x, grad_x = self._basic_gradient_integral(p_, I_, suffix=f'_hs_i{p_}{I_}{suffix}')
            if y_ < self.n_s:
                init_y, grad_y = self._basic_gradient_static_parameter(y_, suffix=f'_hs_sp{y_}{suffix}')
            else:
                p_, I_ = self._mapping_integral[y_ - self.n_s]
                init_y, grad_y = self._basic_gradient_integral(p_, I_, suffix=f'_hs_i{p_}{I_}{suffix}')
            init += init_x + init_y

            for c_x, i_x in grad_x:
                for c_y, i_y in grad_y:
                    row_ = np.repeat(i_x, len(i_y))
                    col_ = np.tile(i_y, len(i_x))
                    if x_ == y_:
                        index, row2_, col2_ = _filter_rc(row_, col_)
                    else:
                        index, row2_, col2_ = _rotate_rc(row_, col_)
                    if np.array_equal(index, np.arange(len(i_x) * len(i_y))):
                        hess.append((f'np.kron({c_x}, {c_y}) * h_hs{suffix}[{row}]', row2_, col2_))
                    elif len(index) > 0:
                        index_str = 'np.array([' + ', '.join(map(str, index)) + '], dtype=np.int32)'
                        hess.append((f'np.kron({c_x}, {c_y})[{index_str}] * h_hs{suffix}[{row}]', row2_, col2_))
        return init, hess

    def objective(self, x):
        v_s = self._basic_value(x)[-1]
        return self.F_o.F(v_s, 1)[0]

    def gradient(self, x):
        Grad = np.zeros_like(x)
        sp, V, SP, MT, DT, T, M, I, v_s = self._basic_value(x)
        init, grad = self._basic_gradient_system('F_o')
        exec(init, globals(), locals())
        for c, i in grad:
            Grad[i] += eval(c)
        return Grad

    def constraints(self, x):
        sp, V, SP, MT, DT, T, M, I, v_s = self._basic_value(x)

        c_system = np.array([f_c.F(v_s, 1)[0] for f_c in self.F_c], dtype=np.float64)

        d_est_phase = np.array([p.D_x @ v[:p.L_x] / dt for p, v, dt in zip(self.p, V, DT)], dtype=np.float64).flatten()
        d_func_phase = np.array([f_d.F(m, p.l0_m) for p, m in zip(self.p, M) for f_d in p.F_d],
                                dtype=np.float64).flatten()
        c_dynamics = d_est_phase - d_func_phase

        c_phase = np.array([f_c.F(m, p.l0_m) for p, m in zip(self.p, M) for f_c in p.F_c], dtype=np.float64).flatten()

        c_boundary = np.array([v_bc - f_bc.F(sp, 1)[0] for p, v in zip(self.p, V)
                               for f_bc, v_bc in zip(p.F_bc, v[p.i_bc])], dtype=np.float64)

        return np.concatenate([c_system, c_dynamics, c_phase, c_boundary])

    def _update_jacobian_structure(self):
        L_c_system = self.n_c
        L_c_dynamics = sum(p.n_x * p.l0_m for p in self.p)
        L_c_phase = sum(p.n_c * p.l0_m for p in self.p)
        L_c_boundary = sum(p.n_c for p in self.p)
        l_c = np.cumsum([0, L_c_system, L_c_dynamics, L_c_phase])

        Jac_row = []
        Jac_col = []
        Jac_code = 'jac_data = np.zeros({})\n'

        # system constraint
        for c_ in range(self.n_c):
            init, grad = self._basic_gradient_system(f'F_c[{c_}]', suffix=f'_c{c_}')
            Jac_code += init
            for c, i in grad:
                Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + len(i)}] = {c}\n'
                Jac_row.extend(np.full_like(i, l_c[0] + c_))
                Jac_col.extend(i)

        c_base_ = l_c[1]
        for p_, p in enumerate(self._phase):
            # dynamics constraint: est
            Jac_code += f'p = self.p[{p_}]\n'
            Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + p.D_coo.nnz}] = p.D_coo.data / DT[{p_}]\n'
            Jac_row += [c_base_ + r for r in p.D_coo.row]
            Jac_col += [self.l_p[p_] + c for c in p.D_coo.col]

            Jac_code += f'grad_t = -(p.D_x @ V[{p_}][:p.L_x]) / DT[{p_}] / DT[{p_}]\n'
            Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + p.n_x * p.l0_m}] = -grad_t\n'
            Jac_row += [c_base_ + r for r in range(p.n_x * p.l0_m)]
            Jac_col += [self.r_p[p_] - 2] * p.n_x * p.l0_m
            Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + p.n_x * p.l0_m}] = grad_t\n'
            Jac_row += [c_base_ + r for r in range(p.n_x * p.l0_m)]
            Jac_col += [self.r_p[p_] - 1] * p.n_x * p.l0_m

            # dynamics constraint: func
            for d_, f_d in enumerate(p.F_d):
                init, grad = self._basic_gradient_phase_func(p_, f'F_d[{d_}]', suffix=f'_p{p_}d{d_}')
                Jac_code += init
                for c, i in grad:
                    Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + p.l0_m}] = -{c}\n'
                    Jac_row.extend(np.arange(c_base_ + d_ * p.l0_m, c_base_ + (d_ + 1) * p.l0_m))
                    Jac_col.extend(i)

            c_base_ += p.n_x * p.l0_m

        # phase constraint
        c_base_ = l_c[2]
        for p_, p in enumerate(self._phase):
            for c_, f_c in enumerate(p.F_c):
                init, grad = self._basic_gradient_phase_func(p_, f'F_c[{c_}]', suffix=f'_p{p_}c{c_}')
                Jac_code += init
                for c, i in grad:
                    Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + p.l0_m}] = {c}\n'
                    Jac_row.extend(np.arange(c_base_ + c_ * p.l0_m, c_base_ + (c_ + 1) * p.l0_m))
                    Jac_col.extend(i)
            c_base_ += p.n_c * p.l0_m

        # boundary constraint
        c_base_ = l_c[3]
        for p_, p in enumerate(self._phase):
            if p.n_bc > 0:
                Jac_code += f'p = self.p[{p_}]\n'
                Jac_code += f'jac_data[{len(Jac_row)}:{len(Jac_row) + p.n_bc}] = np.ones({p.n_bc}, dtype=np.float64)\n'
                Jac_row.extend(range(c_base_, c_base_ + p.n_bc))
                Jac_col.extend(self.l_p[p_] + p.i_bc)
            for bc_, bc in enumerate(p.F_bc):
                Jac_code += f'g_p = p.F_bc[{bc_}].G(sp, 1)\n'
                for row, v_ in enumerate(bc.G_index):
                    Jac_code += f'jac_data[{len(Jac_row)}] = -g_p[{row}]\n'
                    Jac_row.append(c_base_ + bc_)
                    Jac_col.append(self.l_s + v_)
            c_base_ += p.n_bc

        self._jac_row = np.array(Jac_row, dtype=np.int32)
        self._jac_col = np.array(Jac_col, dtype=np.int32)
        self._jac_func_code = Jac_code.format(len(Jac_row))

    def jacobian(self, x):
        sp, V, SP, MT, DT, T, M, I, v_s = self._basic_value(x)
        exec(self._jac_func_code, globals(), locals())
        return locals()['jac_data']

    def jacobianstructure(self):
        return self._jac_row, self._jac_col

    def _update_hessian_structure(self):
        L_c_system = self.n_c
        L_c_dynamics = sum(p.n_x * p.l0_m for p in self.p)
        L_c_phase = sum(p.n_c * p.l0_m for p in self.p)
        L_c_boundary = sum(p.n_c for p in self.p)
        l_c = np.cumsum([0, L_c_system, L_c_dynamics, L_c_phase])

        Hess_row = []
        Hess_col = []
        Hess_code = 'hess_data = np.zeros({})\n'

        # objective
        init_, hess_ = self._basic_hessian_system('F_o')
        Hess_code += init_
        for code, row, col in hess_:
            Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + len(row)}] = {code} * fct_o\n'
            Hess_row.extend(row)
            Hess_col.extend(col)

        # system constraint
        for c_ in range(self.n_c):
            init_, hess_ = self._basic_hessian_system(f'F_c[{c_}]')
            Hess_code += init_
            for code, row, col in hess_:
                Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + len(row)}] = {code} * fct_c[{c_}]\n'
                Hess_row.extend(row)
                Hess_col.extend(col)

        # dynamics constraint
        c_base_ = l_c[1]
        for p_, p in enumerate(self._phase):
            Hess_code += f'p = self.p[{p_}]\n'
            Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + p.D_coo.nnz}] ' \
                         f'= p.D_coo.data / DT[{p_}] / DT[{p_}] * fct_c[{c_base_} + p.D_coo.row]\n'
            Hess_row.extend(np.full_like(p.D_coo.col, self.r_p[p_] - 2))
            Hess_col.extend(self.l_p[p_] + p.D_coo.col)
            Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + p.D_coo.nnz}] ' \
                         f'= -p.D_coo.data / DT[{p_}] / DT[{p_}] * fct_c[{c_base_} + p.D_coo.row]\n'
            Hess_row.extend(np.full_like(p.D_coo.col, self.r_p[p_] - 1))
            Hess_col.extend(self.l_p[p_] + p.D_coo.col)

            Hess_code += f'hess_t = 2 * (p.D_x @ V[{p_}][:p.L_x]) / (DT[{p_}] ** 3) ' \
                         f'* fct_c[{c_base_}:{c_base_ + p.n_x * p.l0_m}]\n'
            Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + p.n_x * p.l0_m}] = hess_t\n'
            Hess_row.extend(np.full(p.n_x * p.l0_m, self.r_p[p_] - 2))
            Hess_col.extend(np.full(p.n_x * p.l0_m, self.r_p[p_] - 2))
            Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + p.n_x * p.l0_m}] = -hess_t\n'
            Hess_row.extend(np.full(p.n_x * p.l0_m, self.r_p[p_] - 1))
            Hess_col.extend(np.full(p.n_x * p.l0_m, self.r_p[p_] - 2))
            Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + p.n_x * p.l0_m}] = hess_t\n'
            Hess_row.extend(np.full(p.n_x * p.l0_m, self.r_p[p_] - 1))
            Hess_col.extend(np.full(p.n_x * p.l0_m, self.r_p[p_] - 1))

            for d_ in range(p.n_x):
                init, hess = self._basic_hessian_phase_func(p_, f'F_d[{d_}]')
                Hess_code += init
                for code, row, col in hess:
                    Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + len(row)}] ' \
                                 f'= -{code} * fct_c[{c_base_ + d_ * p.l0_m}:{c_base_ + (d_ + 1) * p.l0_m}]\n'
                    Hess_row.extend(row)
                    Hess_col.extend(col)

            c_base_ += p.n_x * p.l0_m

        # phase constraint
        c_base_ = l_c[2]
        for p_ in range(self.n_p):
            p = self.p[p_]
            for c_ in range(self.p[p_].n_c):
                init, hess = self._basic_hessian_phase_func(p_, f'F_c[{c_}]')
                Hess_code += init
                for code, row, col in hess:
                    Hess_code += f'hess_data[{len(Hess_row)}:{len(Hess_row) + len(row)}] ' \
                                 f'= {code} * fct_c[{c_base_ + c_ * p.l0_m}:{c_base_ + (c_ + 1) * p.l0_m}]\n'
                    Hess_row.extend(row)
                    Hess_col.extend(col)
            c_base_ += p.n_c * p.l0_m

        # boundary constraint
        c_base_ = l_c[3]
        for p_, p in enumerate(self._phase):
            if p.n_bc > 0:
                Hess_code += f'p = self.p[{p_}]\n'
            for bc_, bc in enumerate(p.F_bc):
                Hess_code += f'h_p = p.F_bc[{bc_}].H(sp, 1)\n'
                for row, (x_, y_) in enumerate(zip(*bc.H_index)):
                    Hess_code += f'hess_data[{len(Hess_row)}] = -h_p[{row}] * fct_c[{c_base_ + bc_}]\n'
                    Hess_row.append(self.l_s + x_)
                    Hess_col.append(self.l_s + y_)
            c_base_ += p.n_bc

        self._hess_row = np.array(Hess_row, dtype=np.int32)
        self._hess_col = np.array(Hess_col, dtype=np.int32)
        self._hess_func_code = Hess_code.format(len(Hess_row))

    def hessianstructure(self):
        return self._hess_row, self._hess_col

    def hessian(self, x, fct_c, fct_o):
        sp, V, SP, MT, DT, T, M, I, v_s = self._basic_value(x)
        exec(self._hess_func_code, globals(), locals())
        return locals()['hess_data']


def _filter_rc(row, col):
    row2 = []
    col2 = []
    index = []
    for i, (r, c) in enumerate(zip(row, col)):
        if r >= c:
            row2.append(r)
            col2.append(c)
            index.append(i)
    return np.array(index, dtype=np.int32), np.array(row2, dtype=np.int32), np.array(col2, dtype=np.int32)


def _rotate_rc(row, col):
    row2 = []
    col2 = []
    index = []
    for i in range(len(row)):
        if row[i] > col[i]:
            row2.append(row[i])
            col2.append(col[i])
            index.append(i)
        elif row[i] == col[i]:
            row2.append(row[i])
            col2.append(col[i])
            index.append(i)
            row2.append(row[i])
            col2.append(col[i])
            index.append(i)
        else:
            row2.append(col[i])
            col2.append(row[i])
            index.append(i)
    return np.array(index, dtype=np.int32), np.array(row2, dtype=np.int32), np.array(col2, dtype=np.int32)
