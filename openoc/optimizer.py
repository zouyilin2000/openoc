from typing import List, Union, Iterable

import cyipopt
import numpy as np

from .system import System
from .variable import Variable


def ipopt_solve(system: System, guess: Union[Variable, List[Union[Variable, Iterable[float]]]],
                optimizer_options: dict = None):
    """Solve the system using IPOPT. If the system has only one phase and no static variables,
     guess can be a single variable. Otherwise, guess should be a list of variables, one for each phase,
     appended with a list of guess values for static variables. Optimizer options should be a dictionary
     of options to pass to Ipopt. See https://coin-or.github.io/Ipopt/OPTIONS.html. Options will be passed
     verbatim to IPOPT.

     Returns the value returned by IPOPT parsed as the same format as guess (a single Variable or a list of
     Variables and static values), and a dictionary of information about the solution returned by IPOPT.
    """
    if not system.ok:
        raise ValueError("system is not fully configured")
    if optimizer_options is None:
        optimizer_options = {}

    guess_is_variable = isinstance(guess, Variable)
    if guess_is_variable:
        guess = [guess]

    x_0 = np.zeros(system.L)
    for i in range(system.n_p):
        x_0[system.l_p[i]:system.r_p[i]] = guess[i].data
    if system.n_s > 0:
        x_0[system.l_s:system.r_s] = np.array(list(guess[-1]), dtype=np.float64)

    solver = cyipopt.Problem(n=int(system.L), m=(len(system.c_lb)), problem_obj=system, lb=system.v_lb, ub=system.v_ub,
                             cl=system.c_lb, cu=system.c_ub)

    for k, v in optimizer_options.items():
        solver.add_option(k, v)

    x, info = solver.solve(x_0)
    result = []
    for i in range(system.n_p):
        result.append(Variable(system.p[i], x[system.l_p[i]:system.r_p[i]]))
    if system.n_s > 0:
        result.append(x[system.l_s:system.r_s])

    if guess_is_variable:
        return result[0], info
    return result, info
