import numpy as np

import cmath

import math

from qiskit import QuantumCircuit, QuantumRegister, BasicAer, transpile
from qiskit.visualization import visualize_transition

from kaleidoscope import bloch_sphere


def cartesian_to_polar(coordinates):
    r_normalized = np.linalg.norm(coordinates)

    if not np.allclose(r_normalized, 1):
        print('Rotation axis ("n" vector) is not normalized.')
        exit(1)

    x = coordinates[0]
    y = coordinates[1]
    z = coordinates[2]

    phi = np.arctan2(y, x)
    theta = np.arccos(z / r_normalized)

    return theta, phi


def polar_to_cartesian(theta, phi):
    return [1 * np.cos(phi) * np.sin(theta), 1 * np.sin(phi) * np.sin(theta), 1 * np.cos(theta)]


def statevector_to_polar(statevector):
    r_1, phi_1 = cmath.polar(statevector[0])
    r_2, phi_2 = cmath.polar(statevector[1])

    theta = 2 * np.arccos(r_1)
    phi = phi_2 - phi_1

    return theta, phi


def init(theta, phi, theta_n, phi_n, alpha):
    size = 1
    register = QuantumRegister(size)
    circuit = QuantumCircuit(register)

    circuit.ry(theta, register)
    circuit.rz(phi, register)

    vector_begin = polar_to_cartesian(theta, phi)
    n = polar_to_cartesian(theta_n, phi_n)

    circuit.rz(-phi_n, register)
    circuit.ry(-theta_n, register)
    circuit.rz(alpha, register)
    circuit.ry(theta_n, register)
    circuit.rz(phi_n, register)

    backend = BasicAer.get_backend('statevector_simulator')
    statevector = backend.run(transpile(circuit, backend)).result().get_statevector(circuit)

    vector_end = polar_to_cartesian(*statevector_to_polar(statevector))

    plot = bloch_sphere([vector_begin, vector_end, n], vectors_color=['#008000', '#0000ff', '#000000'])

    plot.show()

    return statevector, circuit


statevector_, circuit_ = init(math.pi / 3, math.pi / 2, 0.7853, 0, math.pi)
visualize_transition(circuit_, trace=True, fpg=50, spg=2, saveas="bloch_sphere.gif")
