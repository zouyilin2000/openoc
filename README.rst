openoc: free optimal control software
=====================================

Introduction
------------
**Openoc** is a Python package that solves optimal control problems numerically. It utilizes advanced techniques to be powerful, easy to use, and fast.

- **Powerful**: Openoc can solve basically any optimal problem. It employs a multi-phase optimal control model which allows continuous/non-continuous state and control variables, path/integral/boundary constraints, and fixed/free initial and terminal time.

- **Easy to use**: Openoc is designed to be easy to use. It provides a SymPy_-based, intuitive interface for defining and solving problems.

- **Fast**: Openoc is fast. It uses various techniques to speed up the entire compilation & solution process, including symbolic differentiation (with SymPy_), JIT compilation (with Numba_), meta-programming, and more.

Installation
------------
The easiest way to install openoc is using conda_:

1. Install Anaconda_ and create a new environment with **Python 3.9.***

1. Activate the environment, and run in the terminal:

   .. code-block:: console

       conda install numpy scipy sympy numba
       conda install -c conda-forge cyipopt
       pip install openoc

License
-------
MIT. Feel safe to use it in your research & projects.

.. _SymPy: https://www.sympy.org/
.. _Numba: https://numba.pydata.org/
.. _conda: https://conda.io/
.. _Anaconda: https://www.anaconda.com/