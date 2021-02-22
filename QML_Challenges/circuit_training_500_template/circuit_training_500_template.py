#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer,MomentumOptimizer,QNGOptimizer
from pennylane.templates.embeddings import AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers

def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    num_qubits = 3
    num_layers = 2
    batch_size = 5
    steps = 162
    step_size = 0.15
    dev = qml.device("default.qubit", wires=3)
    dev2 = qml.device("default.qubit", wires=3)
    def layer(W):

        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])
     

    def statepreparation(x):
        # print("state preparation x: {}" .format(x))
        qml.BasisState(x, wires=[0, 1, 2])

    @qml.qnode(dev)
    def circuit(weights, x):

        # statepreparation(x)
        AngleEmbedding (x , wires = range ( num_qubits ))
        # print(x)
        for W in weights:
            layer(W)
        # print(qml.expval(qml.PauliZ(0)))
        
        # StronglyEntanglingLayers ( weights , wires = range ( num_qubits ))
        return qml.expval(qml.PauliZ(0))

    # @qml.qnode(dev2)
    # def circuit2(weights, x):

    #     # statepreparation(x)

    #     # for W in weights:
    #     #     layer(W)
    #     AngleEmbedding (x , wires = range ( num_qubits ))
    #     StronglyEntanglingLayers ( weights , wires = range ( num_qubits ))
    #     return qml.expval(qml.PauliY(0))

    def node_caller(weights,x,flag  =False):
        res = (circuit(weights,x) + circuit(weights,x))
        if flag:
            print(res)
        return np.round(res)

    def variational_classifier(var, x,flag  =False):
        # print("variational_classifier:\n weights {}, bias {}, \n x {}".format(var[0],var[1], x ))
        weights = var[0]
        bias = var[1]
        return circuit(weights, x) + bias
      
    def square_loss(labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss


    def accuracy(labels, predictions):

        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss


    def cost(var, X, Y):
        predictions = [(variational_classifier(var, x)) for x in X]
        return square_loss(Y, predictions)



    # Y_train += 1
    # print(Y_train)
    X,Y = X_train, Y_train
    np.random.seed(0)
    

    var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)
    # var_init2 = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

    # print(var_init)

    opt = MomentumOptimizer(step_size)
    # opt2 = MomentumOptimizer(step_size)


    var = var_init
    flag  =False    
    for it in range(steps):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X), (batch_size,))
        X_batch = X[batch_index]
        Y_batch = Y[batch_index]
        var = opt.step(lambda v: cost(v, X_batch, Y_batch), var)
        # var = opt2.step(lambda v: cost(v, X_batch, Y_batch), var)

        # Compute accuracy
        # predictions = [(variational_classifier(var, x,flag)) for x in X]
        # acc = accuracy(Y, np.round(predictions))
        # flag  =False
        # if not(it % 10):
        #     flag  =False
        #     print(array_to_concatenated_string(np.round(predictions)))

        # print(
        #     "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
        #         it + 1, cost(var, X, Y), acc
        #     )
        # )
    # X_test += 1
    predictions = np.round([(variational_classifier(var, x,flag)) for x in X_test])
    # predictions -= 1
    labels_test = [1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]
    # print("Prediction Accuracy : ",accuracy(labels_test,predictions))

    # QHACK #

    return array_to_concatenated_string(np.round(predictions))


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")