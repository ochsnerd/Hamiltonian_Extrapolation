from setuptools import setup

setup(
    name='matrix_extrapolation',
    version='0.1',
    description='Hamiltonian Simulation by Richardson Extrapolation',
    author='David Ochsner',
    author_email='doc@zurich.ibm.com',
    packages=['matrix_extrapolation'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'mpmath', 'qiskit'],  # can I get this directly from requirements.txt?
)
