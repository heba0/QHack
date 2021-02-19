#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    

    def parameter_shift_term(qnode, params, i):
        shifted = params.copy()
        shifted[i] += np.pi/2

        forward = qnode(shifted)  # forward evaluation

        shifted[i] -= np.pi

        backward = qnode(shifted) # backward evaluation

        return 0.5 * (forward - backward)



    def parameter_shift(qnode, params):
        gradients = np.zeros([len(params)])

        for i in range(len(params)):
            gradients[i] = parameter_shift_term(qnode, params, i)

        return gradients


    def Fubini_elem(qnode, params, i,j):

        #elem 1
        shifted = params.copy()
        shifted[i] += np.pi/2
        shifted[j] += np.pi/2
        ket = qnode(shifted)  # forward evaluation
        bra = qnode(params)
        inner_prod_sq1 =  (bra * ket.T)**2


        #elem 2
        shifted = params.copy()
        shifted[i] += np.pi/2
        shifted[j] -= np.pi/2
        ket = qnode(shifted)  # forward evaluation
        bra = qnode(params)
        inner_prod_sq2 =  (bra * ket.T)**2

        #elem 3
        shifted = params.copy()
        shifted[i] -= np.pi/2
        shifted[j] += np.pi/2
        ket = qnode(shifted)  # forward evaluation
        bra = qnode(params)
        inner_prod_sq3 =  (bra * ket.T)**2

        #elem 4
        shifted = params.copy()
        shifted[i] -= np.pi/2
        shifted[j] -= np.pi/2
        ket = qnode(shifted)  # forward evaluation
        bra = qnode(params)
        inner_prod_sq4 =  (bra * ket.T)**2

        return 1/8 * (-inner_prod_sq1+inner_prod_sq2+inner_prod_sq3-inner_prod_sq4)
    
    
    def calc_Fubini(qnode, params):
        F = np.zeros([len(params), len(params)], dtype=np.float64)
        for i in range(len(params)):
            for j in range(len(params)):
                F[i][j] = Fubini_elem(qnode, params, i,j)
        return F


    gradient = parameter_shift(qnode,params)
    F = calc_Fubini(qnode,params)
    F_inv = np.linalg.pinv(F)
    # print(F_inv @ gradient)
    # QHACK #

    return F_inv @ gradient


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
