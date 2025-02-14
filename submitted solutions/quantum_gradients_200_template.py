#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    extra = circuit(weights)*2
    def parameter_shift_term(qnode, params, i):
        shifted = params.copy()
        shifted[i] += np.pi/2
        forward = qnode(shifted)  # forward evaluation
        forward_hessian = forward


        shifted = params.copy()
        shifted[i] -= np.pi/2
        backward = qnode(shifted) # backward evaluation
        backward_hessian = backward


        hessian[i][i] = 0.5*(forward_hessian - extra + backward_hessian)

       # if i == j: 
       #     shifted = params.copy()
       #     shifted[i] += np.pi
       #     first = qnode(shifted)

       #     shifted[i] -= np.pi
       #     second = qnode(shifted)


        return 0.5 * (forward - backward)



    def parameter_shift(qnode, params):
        gradients = np.zeros([len(params)])

        for i in range(len(params)):
            gradients[i] = parameter_shift_term(qnode, params, i)

        return gradients


    def parameter_shift_term_hess(qnode, params, i, j):
        
        if i != j:
            shifted = params.copy()
            shifted[i] += np.pi/2
            shifted[j] += np.pi/2
            first = qnode(shifted)  # forward evaluation
        
            shifted = params.copy()
            shifted[i] -= np.pi/2
            shifted[j] += np.pi/2
            second = qnode(shifted)  # forward evaluation

            shifted = params.copy()
            shifted[i] += np.pi/2
            shifted[j] -= np.pi/2
            third = qnode(shifted) # backward evaluation

            shifted = params.copy()
            shifted[i] -= np.pi/2
            shifted[j] -= np.pi/2
            fourth = qnode(shifted) # backward evaluation

        #if i == j: 
         #   shifted = params.copy()
         #   shifted[i] += np.pi
          #  first = qnode(shifted)

          #  shifted[i] -= np.pi
          #  second = qnode(shifted)

        # return 0.5*(first-second)

        return   0.25*(first - second - third + fourth)




    def calc_hessian(qnode, params):
        # print(params.shape)
        for i in range(5):
            for j in range(i):
                hessian[i][j] = parameter_shift_term_hess(qnode, params, i, j)
                hessian[j][i] = hessian[i][j]
        return hessian



    gradient = parameter_shift(circuit, weights)
    hessian = calc_hessian(circuit, weights)
   
   # print(weights)
   # print(gradient)
    # print (hessian)
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )