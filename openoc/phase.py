from typing import Union, List, Iterable

import sympy as sp

from .discretization import *
from .fastfunc import FastFunc


class Phase:
    def __init__(self, state: Union[int, List[str]], control: Union[int, List[str]],
                 symbol_static_parameter: List[sp.Symbol], identifier: int, /, simplify: bool = False,
                 parallel: bool = False, fastmath: bool = False):
        """Initialize a phase with given state, control and static variables.

        If names are given, they are used as the names of the variables.
        Otherwise, the names are generated automatically as x_0 through x_{n - 1}.

        static variables should be identical with which defined in the system.

        identifier should be unique for each phase in a given system to avoid possible name conflict.

        It is recommended to use method of System object to create phase, instead of manually.

        If simplify is True, every symbolic expression will be simplified (by sympy.simplify)
        before being compiled. This may slow down the speed of compilation.

        If parallel is True, parallel flag will be passed to numba JIT compiler,
        which will generate parallel code for multicore CPUs.
        This may slow down the speed of compilation.

        If fastmath is True, fastmath flag will be passed to numba JIT compiler,
        see numba & LLVM documentations for details.

        Args:
            state (Union[int, List[str]]): Number of state variables or list of state variable names.
            control (Union[int, List[str]]): Number of control variables or list of control variable names.
            symbol_static_parameter (List[sp.Expr]): List of static parameters, should be identical to which in the system.
            identifier (int): Index of the phase.
            simplify (bool, optional): Whether to use sympy to simplify Exprs. Defaults to False.
            parallel (bool, optional): Whether to use numba parallel mode. Defaults to False.
            fastmath (bool, optional): Whether to use numba fastmath mode. Defaults to False.
        """
        self._identifier = identifier

        if isinstance(state, int):
            self._num_state = state
            self._name_state = [f'x_{i}^{{({identifier})}}' for i in range(state)]
        elif isinstance(state, list):
            self._name_state = [s + f'^{{({identifier})}}' for s in state]
            self._num_state = len(state)
        else:
            raise ValueError('state must be int or list of str')

        if isinstance(control, int):
            self._num_control = control
            self._name_control = [f'u_{i}^{{({identifier})}}' for i in range(control)]
        elif isinstance(control, list):
            self._name_control = [c + f'^{{({identifier})}}' for c in control]
            self._num_control = len(control)
        else:
            raise ValueError('control must be int or list of str')

        self._num_static_parameter = len(symbol_static_parameter)
        self._name_static_parameter = [p.name for p in symbol_static_parameter]
        self._symbol_static_parameter = symbol_static_parameter

        self._symbol_state = [sp.Symbol(name) for name in self._name_state]
        self._symbol_control = [sp.Symbol(name) for name in self._name_control]
        self._symbol_time = sp.Symbol(f't^{{({identifier})}}')
        self._symbols = self._symbol_state + self._symbol_control + symbol_static_parameter + [self._symbol_time]

        self._simplify = simplify
        self._parallel = parallel
        self._fastmath = fastmath

        self._dynamics_set = False  # user must set dynamics
        self._boundary_condition_set = False  # user must set boundary conditions
        self._discretization_set = False  # user must set a discretization scheme
        self._integral_set = False
        self._phase_constraint_set = False
        self.set_integral([])  # no integral by default
        self.set_phase_constraint([], [], [])  # no phase constraint by default

    def set_dynamics(self, dynamics: List[Union[float, sp.Expr]]):
        """Set the dynamics of the phase.

        Args:
            dynamics (List[sp.Expr]): List of derivatives of states composed with x, u, s, and t.
        """
        if not len(dynamics) == self._num_state:
            raise ValueError('number of dynamics must be same as number of state variables')

        self._expr_dynamics = [sp.sympify(d) for d in dynamics]
        self._func_dynamics = [FastFunc(d, self._symbols, self._simplify, self._parallel, self._fastmath)
                               for d in self._expr_dynamics]
        self._dynamics_set = True  # must be the last line of this method, to avoid inconsistency if runtime error occurs
        return self

    def set_integral(self, integral: List[Union[float, sp.Expr]]):
        """Set the integrals of the phase.
        Symbols I_0, ..., I_{n - 1} will be automatically generated to be used to represent
        corresponding integrals in system.

        Args:
            integral (List[sp.Expr]): List of integrals to be concerned composed with x, u, s, and t.
        """
        self._expr_integral = [sp.sympify(i) for i in integral]
        self._func_integral = [FastFunc(i, self._symbols, self._simplify, self._parallel, self._fastmath)
                               for i in self._expr_integral]
        self._num_integral = len(integral)
        self._symbol_integral = [sp.Symbol(f'I_{i}^{{({self._identifier})}}') for i in range(self._num_integral)]

        self._integral_set = True
        return self

    def set_phase_constraint(self, phase_constraint: List[Union[float, sp.Expr]],
                             lower_bound: Iterable[Union[float]], upper_bound: Iterable[Union[float]]):
        """Set the phase constraint of the system, which is enforced in the entire phase.
        for equality constraint, set lower bound and upper bound to the same value.

        Args:
            phase_constraint (List[sp.Expr]): List of phase constraints composed with x, u, s, t
        """
        phase_constraint = list(phase_constraint)
        lower_bound = list(lower_bound)
        upper_bound = list(upper_bound)
        if not len(phase_constraint) == len(lower_bound) == len(upper_bound):
            raise ValueError('phase_constraint, lower_bound and upper_bound must have the same length')

        self._variable_bounds_phase = []
        self._static_parameter_bounds_phase = []
        self._time_bounds_phase = []

        self._expr_phase_constraint = []
        lower_bound_phase_constraint = []
        upper_bound_phase_constraint = []
        for c, lb, ub in zip(phase_constraint, lower_bound, upper_bound):
            if c.is_symbol:
                i = self._symbols.index(c)
                if i < self.n:
                    self._variable_bounds_phase.append((i, lb, ub))
                elif i < self.n + self.n_s:
                    self._static_parameter_bounds_phase.append((i - self.n, lb, ub))
                else:
                    self._time_bounds_phase.append((lb, ub))
            else:
                self._expr_phase_constraint.append(sp.sympify(c))
                lower_bound_phase_constraint.append(lb)
                upper_bound_phase_constraint.append(ub)
        self._func_phase_constraint = [FastFunc(c, self._symbols, self._simplify, self._parallel, self._fastmath)
                                       for c in self._expr_phase_constraint]

        self._num_phase_constraint = len(self._expr_phase_constraint)
        self._lower_bound_phase_constraint = np.array(lower_bound_phase_constraint, dtype=np.float64)
        self._upper_bound_phase_constraint = np.array(upper_bound_phase_constraint, dtype=np.float64)

        self._phase_constraint_set = True  # set flag to true **before** calling _update*

        if self._boundary_condition_set and self._discretization_set:
            self._update_bound_variable()

        return self

    def set_boundary_condition(self, initial_value: List[Union[None, float, sp.Expr]],
                               terminal_value: List[Union[None, float, sp.Expr]],
                               initial_time: Union[None, float, sp.Expr],
                               terminal_time: Union[None, float, sp.Expr]):
        """Set the boundary condition & initial/terminal time of the phase. None for free, otherwise fixed.
         Set it to sympy.Expr of static parameters to enforce relations between difference phases.

        Args:
            initial_value (List[Union[None, float]]): List of initial values of states.
            terminal_value (List[Union[None, float]]): List of terminal values of states.
            initial_time (Union[None, float]): Initial time.
            terminal_time (Union[None, float]): Terminal time.
        """
        if not len(initial_value) == len(terminal_value) == self._num_state:
            raise ValueError('initial_value, terminal_value must have the same length as number of state variables')
        for i in range(self.n_x):
            if isinstance(initial_value[i], int):
                initial_value[i] = float(initial_value[i])
            if isinstance(terminal_value[i], int):
                terminal_value[i] = float(terminal_value[i])
        if isinstance(initial_time, int):
            initial_time = float(initial_time)
        if isinstance(terminal_time, int):
            terminal_time = float(terminal_time)

        self._initial_value = initial_value
        self._terminal_value = terminal_value
        self._initial_time = initial_time
        self._terminal_time = terminal_time

        self._expr_boundary_condition = []

        # x0, xf, t0, tf, respectively. records to be used in set_discretization
        self._mapping_boundary_condition = [[], [], False, False]
        for i, v in enumerate(initial_value):
            if isinstance(v, sp.Expr):
                self._mapping_boundary_condition[0].append(i)
                self._expr_boundary_condition.append(v)
        for i, v in enumerate(terminal_value):
            if isinstance(v, sp.Expr):
                self._mapping_boundary_condition[1].append(i)
                self._expr_boundary_condition.append(v)
        if isinstance(initial_time, sp.Expr):
            self._mapping_boundary_condition[2] = True
            self._expr_boundary_condition.append(initial_time)
        if isinstance(terminal_time, sp.Expr):
            self._mapping_boundary_condition[3] = True
            self._expr_boundary_condition.append(terminal_time)

        self._func_boundary_condition = [FastFunc(bc, self._symbol_static_parameter, self._simplify, self._parallel,
                                                  self._fastmath) for bc in self._expr_boundary_condition]

        self._boundary_condition_set = True

        if self._discretization_set:
            self._update_boundary_condition_index()

        if self._discretization_set and self._phase_constraint_set:
            self._update_bound_variable()
        return self

    def set_discretization(self, mesh: Union[int, Iterable[float]], num_point: Union[int, Iterable[int]],
                           state_continuity: Union[bool, Iterable[bool]],
                           control_continuity: Union[bool, Iterable[bool]]):
        """Set the discretization scheme of the system.

        Mesh will be rescaled when needed. If it is set with an integer, uniform mesh is used.
        If it is set with an array, the array will be used as mesh directly.

        num_point decides the number of interpolation / integration points in each mesh interval.
        If it is set with an integer, the same number of points will be used in each mesh interval.

        state_continuous and control_continuous decides whether the state and control are continuous in the phase.
        E.g., in Bang-Bang control problem, set the control_continuous to False for corresponding control variables.
        state & control continuous flags must be correctly set to ensure variables have proper discretization scheme.
        If it is set with a boolean, the same continuity will be used in all states or controls.

        Args:
            mesh (Union[int, NDArray[np.float64]]): Number of mesh or mesh points.
            num_point (Union[int, NDArray[np.int32]]): Number of points in each mesh.
            state_continuity (Union[bool, List[bool]]): List of bools indicating whether the state is continuous or not.
            control_continuity (Union[bool, List[bool]]): List of bools indicating whether the control is continuous or not.
        """
        if isinstance(mesh, int):  # uniform mesh
            self._mesh = np.linspace(0, 1, mesh + 1, endpoint=True)
        else:  # scale to [0, 1]
            mesh = np.array(list(mesh), dtype=np.float64)
            self._mesh = (mesh - mesh[0]) / (mesh[-1] - mesh[0])
        self._num_interval = len(self._mesh) - 1

        if isinstance(num_point, int):
            self._num_point = np.full(self._num_interval, num_point, dtype=np.int32)
        else:
            num_point = np.array(list(num_point), dtype=np.int32)
            self._num_point = num_point

        if len(self._num_point) != self._num_interval:
            raise ValueError('num_point must have the same length as mesh intervals (= len(mesh) - 1)')

        if isinstance(state_continuity, bool):
            self._continuity_state = np.full(self._num_state, state_continuity, dtype=np.bool)
        else:
            self._continuity_state = np.array(list(state_continuity), dtype=np.bool)

        if isinstance(control_continuity, bool):
            self._continuity_control = np.full(self._num_control, control_continuity, dtype=np.bool)
        else:
            self._continuity_control = np.array(list(control_continuity), dtype=np.bool)

        self._continuity = np.concatenate((self._continuity_state, self._continuity_control))

        self.l_c, self.r_c = lr_c(self._num_point)
        self.l_nc, self.r_nc = lr_nc(self._num_point)
        self.L_c = self.r_c[-1]
        self.L_nc = self.r_nc[-1]

        self.t_c, self.w_c = xw_c(self._mesh, self._num_point)
        self.t_nc, self.w_nc = xw_nc(self._mesh, self._num_point)

        self.f_c2nc = c2nc(self._num_point)
        self.f_nc2c = nc2c(self._num_point)
        self.D_c = D_c(self._mesh, self._num_point)
        self.D_nc = D_nc(self._mesh, self._num_point)

        self.l_v, self.r_v = lr_v(self._num_point, self._continuity)
        self.l_m, self.r_m = lr_m(self._num_point, self._continuity)
        self.L_v = self.r_v[-1]
        self.L_m = self.r_m[-1]

        self.f_v2m = v2m(self._num_point, self._continuity)
        self.f_m2v = m2v(self._num_point, self._continuity)
        self.D_x = D_x(self._mesh, self._num_point, self._continuity_state)

        self.D_coo = self.D_x.tocoo()  # cache to speed up jacobian & hessian calculation
        self.D_coo.eliminate_zeros()

        self._discretization_set = True

        if self._boundary_condition_set:
            self._update_boundary_condition_index()
        if self._boundary_condition_set and self._phase_constraint_set:
            self._update_bound_variable()
        return self

    def _update_bound_variable(self):
        """Update lower and upper bound of variables.

        Should be called after discretization scheme, phase constraints, and boundary condition is set.
        """
        self._lower_bound_variable = np.full(self.L, -np.inf, dtype=np.float64)
        self._upper_bound_variable = np.full(self.L, np.inf, dtype=np.float64)
        for i, lb, ub in self._variable_bounds_phase:
            self._lower_bound_variable[self.l_v[i]:self.r_v[i]] \
                = np.maximum(self._lower_bound_variable[self.l_v[i]:self.r_v[i]], lb)
            self._upper_bound_variable[self.l_v[i]:self.r_v[i]] \
                = np.minimum(self._upper_bound_variable[self.l_v[i]:self.r_v[i]], ub)
        for lb, ub in self._time_bounds_phase:
            self._lower_bound_variable[-2] = np.maximum(self._lower_bound_variable[-2], lb)
            self._lower_bound_variable[-1] = np.maximum(self._lower_bound_variable[-1], lb)
            self._upper_bound_variable[-2] = np.minimum(self._upper_bound_variable[-2], ub)
            self._upper_bound_variable[-1] = np.minimum(self._upper_bound_variable[-1], ub)
        for i in range(self.n_x):
            if isinstance(self.bc_0[i], float):
                self._lower_bound_variable[self.l_v[i]] \
                    = np.maximum(self._lower_bound_variable[self.l_v[i]], self.bc_0[i])
                self._upper_bound_variable[self.l_v[i]] \
                    = np.minimum(self._upper_bound_variable[self.l_v[i]], self.bc_0[i])
            if isinstance(self.bc_f[i], float):
                self._lower_bound_variable[self.r_v[i] - 1] \
                    = np.maximum(self._lower_bound_variable[self.r_v[i] - 1], self.bc_f[i])
                self._upper_bound_variable[self.r_v[i] - 1] \
                    = np.minimum(self._upper_bound_variable[self.r_v[i] - 1], self.bc_f[i])
        if isinstance(self.t_0, float):
            self._lower_bound_variable[-2] = np.maximum(self._lower_bound_variable[-2], self.t_0)
            self._upper_bound_variable[-2] = np.minimum(self._upper_bound_variable[-2], self.t_0)
        if isinstance(self.t_f, float):
            self._lower_bound_variable[-1] = np.maximum(self._lower_bound_variable[-1], self.t_f)
            self._upper_bound_variable[-1] = np.minimum(self._upper_bound_variable[-1], self.t_f)

    def _update_boundary_condition_index(self):
        """Update the index of boundary conditions in the discretized system.

        Should be called after both discretization scheme and boundary condition is set.
        """
        index_boundary_condition = []
        for i in self._mapping_boundary_condition[0]:
            index_boundary_condition.append(self.l_v[i])
        for i in self._mapping_boundary_condition[1]:
            index_boundary_condition.append(self.r_v[i] - 1)
        if self._mapping_boundary_condition[2]:
            index_boundary_condition.append(self.L_v)
        if self._mapping_boundary_condition[3]:
            index_boundary_condition.append(self.L_v + 1)
        self._index_boundary_condition = np.array(index_boundary_condition, dtype=np.int32)
        self._num_boundary_condition = len(self._index_boundary_condition)

    @property
    def n_x(self) -> int:
        """Number of states."""
        return self._num_state

    @property
    def x(self) -> List[sp.Symbol]:
        """Sympy symbols of states."""
        return self._symbol_state

    @property
    def c_x(self) -> NDArray[np.bool]:
        """Continuity of states."""
        return self._continuity_state

    @property
    def n_u(self) -> int:
        """Number of controls."""
        return self._num_control

    @property
    def u(self) -> List[sp.Symbol]:
        """Sympy symbols of controls."""
        return self._symbol_control

    @property
    def c_u(self) -> NDArray[np.bool]:
        """Continuity of controls."""
        return self._continuity_control

    @property
    def n(self) -> int:
        """Number of states and controls."""
        return self._num_state + self._num_control

    @property
    def c(self) -> NDArray[np.bool]:
        """Continuity of states and controls."""
        return self._continuity

    @property
    def n_s(self) -> int:
        """Number of static parameters."""
        return self._num_static_parameter

    @property
    def s(self) -> List[sp.Symbol]:
        """Sympy symbols of static parameters."""
        return self._symbol_static_parameter

    @property
    def t(self):
        """Sympy symbols of time."""
        return self._symbol_time

    @property
    def F_d(self) -> List[FastFunc]:
        """FastFuncs of dynamics."""
        return self._func_dynamics

    @property
    def n_d(self) -> int:
        """Number of dynamics."""
        return self._num_state

    @property
    def F_I(self) -> List[FastFunc]:
        """FastFuncs of integrals."""
        return self._func_integral

    @property
    def n_I(self) -> int:
        """Number of integrals."""
        return self._num_integral

    @property
    def I(self) -> List[sp.Symbol]:
        """Sympy symbols of integrals."""
        return self._symbol_integral

    @property
    def F_c(self) -> List[FastFunc]:
        """FastFuncs of constraints."""
        return self._func_phase_constraint

    @property
    def n_c(self) -> int:
        """Number of constraints."""
        return self._num_phase_constraint

    @property
    def v_lb(self) -> NDArray[np.float64]:
        """Lower bounds of variables."""
        return self._lower_bound_variable

    @property
    def v_ub(self) -> NDArray[np.float64]:
        """Upper bounds of variables."""
        return self._upper_bound_variable

    @property
    def c_lb(self) -> NDArray[np.float64]:
        """Lower bounds of constraints."""
        return self._lower_bound_phase_constraint

    @property
    def c_ub(self) -> NDArray[np.float64]:
        """Upper bounds of constraints."""
        return self._upper_bound_phase_constraint

    @property
    def s_b(self) -> List[Tuple[int, float, float]]:
        """Bounds of static parameters."""
        return self._static_parameter_bounds_phase

    @property
    def F_bc(self) -> List[FastFunc]:
        """FastFuncs of boundary conditions which is sympy.expr."""
        return self._func_boundary_condition

    @property
    def n_bc(self) -> int:
        """Number of boundary conditions which is sympy.expr."""
        return self._num_boundary_condition

    @property
    def i_bc(self) -> NDArray[np.float64]:
        """Index of boundary conditions which is sympy.expr."""
        return self._index_boundary_condition

    @property
    def bc_0(self) -> List[Union[None, int, float, sp.Expr]]:
        """Initial boundary conditions."""
        return self._initial_value

    @property
    def bc_f(self) -> List[Union[None, int, float, sp.Expr]]:
        """Terminal boundary conditions."""
        return self._terminal_value

    @property
    def t_0(self) -> Union[None, int, float, sp.Expr]:
        """Initial time."""
        return self._initial_time

    @property
    def t_f(self) -> Union[None, int, float, sp.Expr]:
        """Terminal time."""
        return self._terminal_time

    @property
    def N(self) -> int:
        """Number of mesh intervals."""
        return self._num_interval

    @property
    def L(self) -> int:
        """Length of optimization variables, including states, controls, and time."""
        return self.r_v[-1] + 2

    @property
    def L_x(self) -> int:
        """Length of optimization variables of states."""
        return self.r_v[self.n_x - 1]

    @property
    def L_xu(self) -> int:
        """Length of optimization variables of states and controls."""
        return self.r_v[-1]

    @property
    def l0_m(self) -> int:
        """Length of single variable in middle stage"""
        return self.L_nc

    @property
    def t_m(self) -> NDArray[np.float64]:
        """t in middle stage"""
        return self.t_nc

    @property
    def w(self) -> NDArray[np.float64]:
        """Integration weights is integration weights for non-continuous variables."""
        return self.w_nc

    @property
    def ok(self):
        """Whether the phase is fully configured."""
        return self._dynamics_set and self._boundary_condition_set and self._discretization_set
