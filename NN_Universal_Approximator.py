import numpy as np
import pandas as pd


def BackProp(x_train, y_train, nn_architecture, alpha, zeta, x0, max_epochs, tolerance, cost_ftn, Weights_list=None,
             bias_list=None):
    Q = len(x_train)

    if not Weights_list:
        Weights_list = []
        for i in range(len(nn_architecture) - 1):
            Weights_list.append(np.random.uniform(-zeta, zeta, (nn_architecture[i], \
                                                                nn_architecture[i + 1])))
    if not bias_list:
        bias_list = []
        for i in range(len(nn_architecture) - 1):
            bias_list.append(np.random.uniform(-zeta, zeta, (1, nn_architecture[i + 1])))

    L = len(Weights_list)

    for epoch in range(max_epochs):
        epoch_error = 0
        for iteration in range(Q):
            x = x_train[iteration]
            y = y_train[iteration]
            x = np.array(x).reshape((1, len(x)))
            y = np.array(y).reshape((1, len(y)))

            # Calculate activations for all layers
            # don't need to save n_l if we are using bipolar sigmoid transfer function
            a_l_list = [x]
            for i in range(len(Weights_list)):
                n_l = np.matmul(a_l_list[-1], Weights_list[i]) + bias_list[i]
                a_l = transfer_ftn(n_l, x0)
                a_l_list.append(a_l)

            # calculating the error for this example
            y_in = a_l_list[-1]  # activation of the last layer
            example_error = np.matmul(y_in - y, (y_in - y).T)
            example_error = np.asscalar(example_error)
            epoch_error = epoch_error + example_error

            # Calculate sensitivities for last layer. Performs element-wise multiplication.
            # TODO check if this is the only change that needs to be made when using cross-entropy cost function.
            if cost_ftn == "quadratic":
                s_L = np.multiply((y_in - y), derivative_transfer_ftn(y_in, x0))
            elif cost_ftn == "cross-entropy":
                s_L = y_in - y

            # Calculate sensitivites for other layers
            sensitivities_list = [s_L]

            for l in range(L - 1, 0, -1):
                s_l = np.multiply(np.matmul(sensitivities_list[0], Weights_list[l].T), \
                                  derivative_transfer_ftn(a_l_list[l], x0))
                sensitivities_list.insert(0, s_l)

            # Update weights and biases
            for l in range(L):  # L is length of Weights_list
                Weights_list[l] = Weights_list[l] - \
                                  (alpha * np.matmul(a_l_list[l].T, sensitivities_list[l]))

                bias_list[l] = bias_list[l] - (alpha * sensitivities_list[l])

        # epoch error is not normalized (divided by number of examples)
        if epoch_error < tolerance:
            break

    num_training_epochs = epoch + 1
    if num_training_epochs < max_epochs:
        convergence = True
    else:
        convergence = False

    return Weights_list, bias_list, num_training_epochs, epoch_error, convergence


def transfer_ftn(n_l, x0):
    a_l = np.tanh(n_l / (2 * x0))
    return a_l


def derivative_transfer_ftn(a_l, x0):  # we only save a_l NOT n_l if using bipolar sigmoid transfer function
    derivative = ((1 + a_l) * (1 - a_l)) / (2 * x0)
    return derivative


def test_transfer_ftn(y_in):
    if y_in >= 0:
        y_hat = 1
    else:
        y_hat = -1

    return y_hat


def Part1_weights_validation():
    cost_ftn = "quadratic"
    x_train = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y_train = [[-1], [1], [1], [-1]]
    nn_architecture = [2, 4, 1]
    alpha = 0.2  # learning rate
    zeta = 1  # doesn't really matter since we are passing in weights_list and bias_list
    x0 = 1
    max_epochs = 1
    tolerance = 0.05

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

    Weights_list, bias_list, num_training_epochs, epoch_error, convergence = \
        BackProp(x_train, y_train, nn_architecture, alpha, zeta, x0, max_epochs, tolerance, cost_ftn, Weights_list,
                 bias_list)

    print("W1=", Weights_list[0], sep="\n")
    print("b1=", bias_list[0], sep="\n")
    print("W2=", Weights_list[1], sep="\n")
    print("b2=", bias_list[1], sep="\n")
    print()

    np.savez('Part1_results.npz', W1=Weights_list[0], b1=bias_list[0], W2=Weights_list[1], b2=bias_list[1])

    # results can be loaded from the Part1_results.npz file by uncommenting the following


#     data = np.load('Part1_results.npz')
#     W1 = data['W1']
#     b1 = data['b1']
#     W2 = data['W2']
#     b2 = data['b2']


def Part_2a(cost_ftn):
    x_train = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y_train = [[-1], [1], [1], [-1]]
    nn_architecture = [2, 4, 1]  # N1 = 4
    alpha_list = [0.1, 0.2, 0.3]  # learning rate
    zeta_list = [0.5, 1, 1.5]
    x0_list = [0.5, 1, 1.5]
    max_epochs = 500  # this value was chosen empirically after trying several different settings.
    tolerance = 0.05

    # Try all 3X3X3=27 hyperparameter combinations of alpha, zeta and x0
    num_convergence = 0
    for alpha in alpha_list:
        for zeta in zeta_list:
            for x0 in x0_list:
                Weights_list, bias_list, num_training_epochs, epoch_error, convergence = \
                    BackProp(x_train, y_train, nn_architecture, alpha, zeta, x0, max_epochs, tolerance, cost_ftn)
                if convergence:
                    num_convergence += 1
                print("-------------------------------------------------------------------------------")
                print(f"Learning rate = {alpha}, Zeta = {zeta}, x0 = {x0}")
                print(
                    f"Convergence = {convergence}, Training Epochs = {num_training_epochs}, Squared Error = {epoch_error}")
    print("-------------------------------------------------------------------------------")
    print(f"Number of convergent hyperparameter combinations = {num_convergence} (out of 27)")

    # TODO Look for patterns when do we get non-convergent results


def Part_2b(cost_ftn):
    x_train = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y_train = [[-1], [1], [1], [-1]]
    N1_list = [1, 2, 4, 6, 8, 10]
    alpha = 0.2  # learning rate
    zeta = 1
    x0 = 1
    max_epochs = 500  # this value was chosen empirically after trying several different settings.
    tolerance = 0.05

    # Try all 5 hyperparameter combinations of N1
    num_convergence_list = []
    for N1 in N1_list:
        nn_architecture = [2, N1, 1]
        num_convergence = 0
        for iters in range(100):
            Weights_list, bias_list, num_training_epochs, epoch_error, convergence = \
                BackProp(x_train, y_train, nn_architecture, alpha, zeta, x0, max_epochs, tolerance, cost_ftn)
            if convergence:
                num_convergence += 1
        num_convergence_list.append(num_convergence)
    print(f"Convergence results for N1 = [1,2,4,6,8,10] (out of 100): {num_convergence_list}")

    # Results mostly converge for N1=4 and above. For N1=2, almost 70% of the times, it converges. For N1=1, it doesn't converge at all.
    # This is probably because the XOR problem is not linearly separable and we need a higher number of neurons in the hidden layer to approximate the function (see universality theorem).


def main():
    test_transfer_function = np.vectorize(test_transfer_ftn)  # vectorizes the transfer function used at test time
    print("Part 1 Weights Validation:\n")
    Part1_weights_validation()

    # Run the 2 sets of experiments with quadratic cost function
    cost_ftn = "quadratic"
    Part_2a(cost_ftn)
    Part_2b(cost_ftn)

    # Run the 2 sets of experiments with bipolar cross-entropy cost function
    cost_ftn = "cross-entropy"
    Part_2a(cost_ftn)
    Part_2b(cost_ftn)

if __name__ == "__main__":
    main()