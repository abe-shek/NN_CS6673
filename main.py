import getopt
import sys
import numpy as np

from model import Model


def transfer_ftn(n_l, x0):
    a_l = np.tanh(n_l / (2 * x0))
    return a_l


# we only save a_l NOT n_l if using bipolar sigmoid transfer function
def derivative_transfer_ftn(a_l, x0):
    derivative = ((1 + a_l) * (1 - a_l)) / (2 * x0)
    return derivative


def init_weights_biases(model):
    if model.weights_1.all() is 0:
        model.weights_1 = np.random.uniform(-1 * model.hyper_params.zeta,
                                            model.hyper_params.zeta, model.weights_1.shape)
    if model.weights_2.all() is 0:
        model.weights_2 = np.random.uniform(-1 * model.hyper_params.zeta,
                                            model.hyper_params.zeta, model.weights_1.shape)
    if model.biases_1.all() is 0:
        model.biases_1 = np.random.uniform(1 * model.hyper_params.zeta,
                                           model.hyper_params.zeta, model.biases_1.shape)
    if model.biases_2.all() is 0:
        model.biases_2 = np.random.uniform(1 * model.hyper_params.zeta,
                                           model.hyper_params.zeta, model.biases_2.shape)

    return [model.weights_1, model.weights_2], [model.biases_1, model.biases_2]


def train_nn(x_train, y_train, model):
    Q = len(x_train)
    weight_list, bias_list = init_weights_biases(model)
    weight_list_len = len(weight_list)
    for epoch in range(model.hyper_params.max_epochs):
        epoch_error = 0
        for iteration in range(Q):
            x = x_train[iteration]
            y = y_train[iteration]
            x = np.array(x).reshape((1, len(x)))
            y = np.array(y).reshape((1, len(y)))

            # Calculate activations for all layers
            # don't need to save n_l if we are using bipolar sigmoid transfer function
            a_l_list = [x]
            for i in range(len(weight_list)):
                n_l = np.matmul(a_l_list[-1], weight_list[i]) + bias_list[i]
                a_l = transfer_ftn(n_l, model.hyper_params.x0)
                a_l_list.append(a_l)

            # calculating the error for this example
            y_in = a_l_list[-1]  # activation of the last layer
            example_error = np.matmul(y_in - y, (y_in - y).T)
            example_error = np.asscalar(example_error)
            epoch_error = epoch_error + example_error

            # Calculate sensitivities for last layer. Performs element-wise multiplication.

            # TODO - update
            # check if this is the only change that needs to be made
            # when using cross-entropy cost function.
            if model.hyper_params.cost_fn == 0:
                s_L = np.multiply((y_in - y), derivative_transfer_ftn(y_in, model.hyper_params.x0))
            elif model.hyper_params.cost_fn == 1:
                s_L = y_in - y

            # Calculate sensitivites for other layers
            sensitivities_list = [s_L]

            for l in range(weight_list_len - 1, 0, -1):
                s_l = np.multiply(np.matmul(sensitivities_list[0], weight_list[l].T), \
                                  derivative_transfer_ftn(a_l_list[l], model.hyper_params.x0))
                sensitivities_list.insert(0, s_l)

            # Update weights and biases
            for l in range(weight_list_len):
                weight_list[l] = weight_list[l] - \
                                 (model.hyper_params.learning_rate *
                                  np.matmul(a_l_list[l].T, sensitivities_list[l]))

                bias_list[l] = bias_list[l] - \
                               (model.hyper_params.learning_rate * sensitivities_list[l])

        # epoch error is not normalized (divided by number of examples)
        if epoch_error < model.hyper_params.tolerance:
            break

    num_training_epochs = epoch + 1
    if num_training_epochs < model.hyper_params.max_epochs:
        convergence = True
    else:
        convergence = False

    model.weights_1 = weight_list[0]
    model.weights_2 = weight_list[1]
    model.biases_1 = bias_list[0]
    model.biases_2 = bias_list[1]
    return model, num_training_epochs, epoch_error, convergence


def xor_weight_validation(x_train, y_train, model):
    model.hyper_params.max_epochs = 1

    # Setting initial weights and biases for xor weight validation
    model.weights_1 = np.array([[0.197, 0.3191, -0.1448, 0.3594],
                                [0.3099, 0.1904, -0.0347, -0.4861]]).reshape(model.weights_1.shape)
    model.weights_2 = np.array([0.4919, -0.2913, -0.3979, 0.3581]).reshape(model.weights_2.shape)
    model.biases_1 = np.array([-0.3378, 0.2771, 0.2859, -0.3329]).reshape(model.biases_1.shape)
    model.biases_2 = np.array([-0.1401]).reshape(model.biases_2.shape)

    model, num_training_epochs, epoch_error, convergence = train_nn(x_train, y_train, model)

    print("W1=", model.weights_1, sep="\n")
    print("b1=", model.biases_1, sep="\n")
    print("W2=", model.weights_2, sep="\n")
    print("b2=", model.biases_2, sep="\n")

    np.savez('xor_weight_validation.npz', model=model)

    # results can be loaded from the xor_weight_validation.npz file by uncommenting the following
    # data = np.load('Part1_results.npz')
    # model = data['model']


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["steps=", "alpha=", "asch=", "perc="])
    except getopt.GetoptError:
        print("Use - python main.py --steps <num_of_training_steps> --alpha <alpha> "
              "--asch <alpha_scheduling_option> --perc <f_perc_value>")
        sys.exit(2)
    model = Model(2, 4, 1)
    # model.hyper_params = HyperParams()
    for opt, arg in opts:
        if opt == "--steps":
            model.hyper_params.training_steps = int(arg)
        elif opt == "--alpha":
            model.hyper_params.learning_rate = float(arg)
        elif opt == "--asch":
            model.hyper_params.lr_scheduling_option = int(arg)
        elif opt == "--perc":
            model.hyper_params.lr_perc_decrease = float(arg)

    x_train = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y_train = [[-1], [1], [1], [-1]]

    xor_weight_validation(x_train, y_train, model)


if __name__ == "__main__":
    main(sys.argv[1:])
