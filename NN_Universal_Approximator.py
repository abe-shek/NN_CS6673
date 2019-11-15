import numpy as np
import pandas as pd


def BackPropPart1(x_train, y_train, num_neurons_list, alpha, num_iters, Weights_list=None, bias_list=None):
    Q = len(x_train)

    if not Weights_list:
        Weights_list = []
        for i in range(len(num_neurons_list) - 1):
            Weights_list.append(np.random.uniform(-0.5, 0.5, (num_neurons_list[i], \
                                                              num_neurons_list[i + 1])))
    if not bias_list:
        bias_list = []
        for i in range(len(num_neurons_list) - 1):
            bias_list.append(np.random.uniform(-0.5, 0.5, (1, num_neurons_list[i + 1])))


    L = len(Weights_list)

    for i in range(num_iters):
        x = x_train[i % Q]
        y = y_train[i % Q]
        x = np.array(x).reshape((1, len(x)))
        y = np.array(y).reshape((1, len(y)))

        # Calculate activations for all layers
        # don't need to save n_l since using bipolar sigmoid transfer function
        a_l_list = [x]
        for i in range(len(Weights_list)):
            n_l = np.matmul(a_l_list[-1], Weights_list[i]) + bias_list[i]
            a_l = transfer_ftn(n_l)
            a_l_list.append(a_l)

        # Calculate sensitivities for last layer. Performs element-wise multiplication.
        s_L = np.multiply((a_l_list[-1] - y), derivative_transfer_ftn(a_l_list[-1]))

        # Calculate sensitivites for other layers
        sensitivities_list = [s_L]

        for l in range(L - 1, 0, -1):
            s_l = np.multiply(np.matmul(sensitivities_list[0], Weights_list[l].T), \
                              derivative_transfer_ftn(a_l_list[l]))
            sensitivities_list.insert(0, s_l)

        # Update weights and biases
        for l in range(L):  # L is length of Weights_list
            Weights_list[l] = Weights_list[l] - \
                              (alpha * np.matmul(a_l_list[l].T, sensitivities_list[l]))

            bias_list[l] = bias_list[l] - (alpha * sensitivities_list[l])

    #     print("Sensitivities =", sensitivities_list)
    #     print("Activations =", a_l_list)

    # testing if all examples are classified correctly
    num_correct = 0
    #     total_training_error = 0
    total_testing_error = 0
    for q in range(Q):
        x = x_train[q]
        y = y_train[q]
        x = np.array(x).reshape((1, len(x)))
        y = np.array(y).reshape((1, len(y)))

        # Calculate activations for all layers
        a_l_list = [x]
        for i in range(len(Weights_list)):
            n_l = np.matmul(a_l_list[-1], Weights_list[i]) + bias_list[i]
            a_l = transfer_ftn(n_l)
            a_l_list.append(a_l)

        y_in = a_l_list[-1]
        y_hat = test_transfer_function(y_in)

        if np.array_equal(y_hat, y):
            num_correct += 1

        #         training_error = np.matmul(y_in - y, (y_in - y).T)
        #         training_error = np.asscalar(training_error)

        testing_error = np.matmul(y_hat - y, (y_hat - y).T)
        testing_error = np.asscalar(testing_error)

        #         total_training_error = total_training_error + training_error
        total_testing_error = total_testing_error + testing_error

    #     training_mse = total_training_error/Q
    testing_mse = total_testing_error / Q

    print("Number of correct classifications =", num_correct)
    if num_correct == Q:
        print("All examples were correctly classifed.")
    else:
        print("Not all examples were correctly classified.")

    #     print("Weights =", Weights_list)
    #     print("Biases =", bias_list)
    #     print("Final Alpha =", alpha)
    #     print("Number of epochs =", num_iters/Q)
    # #     print("Training mse =", training_mse)
    #     print("Testing mse =", testing_mse)
    #     print()

    return Weights_list, bias_list, testing_mse


def transfer_ftn(n_l):
    a_l = np.tanh(n_l / 2)
    return a_l


def derivative_transfer_ftn(a_l):  # we only save a_l NOT n_l
    derivative = ((1 + a_l) * (1 - a_l)) / 2
    return derivative


def test_transfer_ftn(y_in):
    if y_in >= 0:
        y_hat = 1
    else:
        y_hat = -1

    return y_hat


def main():
    x_train = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y_train = [[-1], [1], [1], [-1]]
    num_neurons_list = [2, 4, 1]
    num_iters = 4  # For Part 1.
    lr = 0.2  # For Part 1.

    # Setting weights and biases for Part 1
    W1 = [[0.197, 0.3191, -0.1448, 0.3594], [0.3099, 0.1904, -0.0347, -0.4861]]
    W1 = np.array(W1).reshape((2, 4))
    b1 = [-0.3378, 0.2771, 0.2859, -0.3329]
    b1 = np.array(b1).reshape((1, 4))
    W2 = [0.4919, -0.2913, -0.3979, 0.3581]
    W2 = np.array(W2).reshape((4, 1))
    b2 = [-0.1401]
    b2 = np.array(b2).reshape((1, 1))

    Weights_list = [W1, W2]
    bias_list = [b1, b2]

    Weights_list, bias_list, testing_mse = \
        BackPropPart1(x_train, y_train, num_neurons_list, lr, \
                      num_iters, Weights_list, bias_list)

    print("W1=", Weights_list[0], sep="\n")
    print("b1=", bias_list[0], sep="\n")
    print("W2=", Weights_list[1], sep="\n")
    print("b2=", bias_list[1], sep="\n")


test_transfer_function = np.vectorize(test_transfer_ftn)
main()