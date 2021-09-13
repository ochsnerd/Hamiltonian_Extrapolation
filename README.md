Richardson Extrapolation for Hamiltonian Simulation
===================================================

> Simulating the time evolution of quantum systems is an application of quantum computing
> of great importance to many fields such as chemistry, physics and optimization.
> Recent works [Carrera2020](https://arxiv.org/abs/2009.04484) apply extrapolation techniques to product formulas
> with promising performance in practical problems. The analytical quantification of the error
> of such methods is a challenging problem. In this thesis, we develop a series expansion expression for the
> extrapolation error based on the Baker-Campbell-Hausdorff formula, together with a python algorithm to
> compute the error by truncating this series. The complexity of the algorithm is analyzed
> and practical aspects of the implementation are discussed. Finally, the approximation obtained from this
> algorithm is showcased on an example system.

This is the code developed as part of my Master's thesis at ETH Zurich & IBM Research under the supervision of Almudena Carrera Vazquez, Stefan Wörner and Prof. Ralph Hiptmair.

´matrix_extrapolation´ - python module; contains a framework for Hamiltonian simulation using product formulas & extrapolation, based on ´numpy´ and ´mpmath´, as well as the extrapolation error series expansion algorithm. Install it by running
´´´
pip install path/to/this/repo
´´´

´scripts´ - scripts showcasing the usage of the module. Used to create the plots for my thesis report.
