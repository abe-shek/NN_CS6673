import getopt
import sys
import numpy as np

from model import Model, HyperParams
from save import save_data, export_data

export_to_excel = False


def transfer_ftn(n_l, x0):
    a_l = np.tanh(n_l / (2 * x0))
    return a_l


# we only save a_l NOT n_l if using bipolar sigmoid transfer function
def derivative_transfer_ftn(a_l, x0):
    derivative = ((1 + a_l) * (1 - a_l)) / (2 * x0)
    return derivative


def init_weights_biases(model):
    if model.reinitialize_weights:
        model.weights_1 = np.random.uniform(-1 * model.hyper_params.zeta,
                                            model.hyper_params.zeta, model.weights_1.shape)

        model.weights_2 = np.random.uniform(-1 * model.hyper_params.zeta,
                                            model.hyper_params.zeta, model.weights_2.shape)

        model.biases_1 = np.random.uniform(-1 * model.hyper_params.zeta,
                                           model.hyper_params.zeta, model.biases_1.shape)

        model.biases_2 = np.random.uniform(-1 * model.hyper_params.zeta,
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
            y_hat = a_l_list[-1]  # activation of the last layer
            example_error = np.matmul(y_hat - y, (y_hat - y).T)
            example_error = np.asscalar(example_error)
            epoch_error = epoch_error + example_error

            # Calculate sensitivities for last layer. Performs element-wise multiplication.
            # quadratic cost function
            if model.hyper_params.cost_fn == 0:
                s_L = np.multiply((y_hat - y), derivative_transfer_ftn(y_hat, model.hyper_params.x0))
            # cross entropy cost function
            elif model.hyper_params.cost_fn == 1:
                s_L = y_hat - y

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

        # epoch error is not normalized (not divided by number of examples)
        if epoch_error < model.hyper_params.tolerance:
            break

    num_training_epochs = epoch + 1
    if num_training_epochs < model.hyper_params.max_epochs:
        convergence = True
    else:
        convergence = False

    update_model_info(model, weight_list, bias_list, num_training_epochs, epoch_error, convergence)

    return model


def update_model_info(model, weight_list, bias_list, num_training_epochs, epoch_error, convergence):
    model.weights_1 = weight_list[0]
    model.weights_2 = weight_list[1]
    model.biases_1 = bias_list[0]
    model.biases_2 = bias_list[1]
    model.model_info.total_epochs_req = num_training_epochs
    model.model_info.last_epoch_error = epoch_error
    model.model_info.converged = convergence


def extract_model_info(model, sheet_name, verbose=True):
    if verbose:
        print("--------------------------------------------------------------------------------")
        print(f"Learning rate = {model.hyper_params.learning_rate} | "
              f"Zeta = {model.hyper_params.zeta} | "
              f"x0 = {model.hyper_params.x0}")
        print(f"Convergence = {model.model_info.converged} | "
              f"Training Epochs = {model.model_info.total_epochs_req} | "
              f"Squared Error = {model.model_info.last_epoch_error}")
        print("--------------------------------------------------------------------------------")

    if export_to_excel:
        save_data(sheet_name, model)


def part_2a(x_train, y_train, model, sheet_name):
    # TODO
    # Look for patterns when do we get non-convergent results
    # Try all 3X3X3=27 hyper parameter combinations of alpha, zeta and x0
    num_convergence = 0
    for alpha in model.hyper_params.alpha_list:
        for zeta in model.hyper_params.zeta_list:
            for x0 in model.hyper_params.x0_list:
                model.hyper_params.learning_rate = alpha
                model.hyper_params.zeta = zeta
                model.hyper_params.x0 = x0
                model = train_nn(x_train, y_train, model)
                if model.model_info.converged:
                    num_convergence += 1
                extract_model_info(model, sheet_name, verbose=True)
    print("-----------------------------------------------------------------------")
    print(f"Number of convergent hyper parameter combinations = {num_convergence} (out of 27)")


def part_2b(x_train, y_train, cost_fn, sheet_name):
    N1_list = [1, 2, 4, 6, 8, 10]
    convergence_list = []
    for i in range(len(N1_list)):
        model = Model(2, N1_list[i], 1)
        model.hyper_params.cost_fn = cost_fn
        num_convergence = 0
        for iters in range(100):
            model = train_nn(x_train, y_train, model)
            if model.model_info.converged:
                num_convergence += 1
            extract_model_info(model, sheet_name, verbose=False)
        convergence_list.append(num_convergence)
        print(f"Convergence for N1 = %d -> %d" % (N1_list[i], num_convergence))

    print(f"Convergence results for N1 = [1,2,4,6,8,10] (out of 100): {convergence_list}")

    # Results mostly converge for N1=4 and above. For N1=2, almost 70% of the times,
    # it converges. For N1=1, it doesn't converge at all.
    # This is probably because the XOR problem is not linearly separable and we need a higher
    # number of neurons in the hidden layer to approximate the function (see universality theorem).


def xor_weight_validation(x_train, y_train, model, sheet_name):
    model.hyper_params.max_epochs = 1

    # Setting initial weights and biases for xor weight validation
    model.weights_1 = np.array([[0.197, 0.3191, -0.1448, 0.3594],
                                [0.3099, 0.1904, -0.0347, -0.4861]]).reshape(model.weights_1.shape)
    model.weights_2 = np.array([0.4919, -0.2913, -0.3979, 0.3581]).reshape(model.weights_2.shape)
    model.biases_1 = np.array([-0.3378, 0.2771, 0.2859, -0.3329]).reshape(model.biases_1.shape)
    model.biases_2 = np.array([-0.1401]).reshape(model.biases_2.shape)
    model.reinitialize_weights = False

    model = train_nn(x_train, y_train, model)

    print("W1=", model.weights_1, sep="\n")
    print("b1=", model.biases_1, sep="\n")
    print("W2=", model.weights_2, sep="\n")
    print("b2=", model.biases_2, sep="\n")

    extract_model_info(model, sheet_name, verbose=False)

    # np.savez('xor_weight_validation.npz', model=model)
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
    hyper_params = HyperParams()
    for opt, arg in opts:
        if opt == "--steps":
            hyper_params.training_steps = int(arg)
        elif opt == "--alpha":
            hyper_params.learning_rate = float(arg)
        elif opt == "--asch":
            hyper_params.lr_scheduling_option = int(arg)
        elif opt == "--perc":
            hyper_params.lr_perc_decrease = float(arg)

    # # uncomment this if data needs to be stored in excel
    # global export_to_excel
    # export_to_excel = True

    x_train = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    y_train = [[-1], [1], [1], [-1]]

    model = Model(2, 4, 1)
    model.hyper_params = hyper_params
    xor_weight_validation(x_train, y_train, model, sheet_name="XOR weights validation")

    # using quadratic cost ftn
    model = Model(2, 4, 1)
    model.hyper_params.cost_fn = 0
    part_2a(x_train, y_train, model, sheet_name="A-Z-X0 variations (Quad)")
    part_2b(x_train, y_train, cost_fn=0, sheet_name="N1 variations (Quad)")

    # using cross entropy cost ftn
    model = Model(2, 4, 1)
    model.hyper_params.cost_fn = 1
    part_2a(x_train, y_train, model, sheet_name="A-Z-X0 variations (CrsEnt)")
    part_2b(x_train, y_train, cost_fn=1, sheet_name="N1 variations (CrsEnt)")

    model = Model(2, 4, 1)
    model.hyper_params.learning_rate = 0.2
    model.hyper_params.zeta = 1.0
    model.hyper_params.x0 = 1.0
    model.hyper_params.cost_fn = 1
    model.hyper_params.max_epochs = 1
    model = train_nn(x_train, y_train, model)
    extract_model_info(model, sheet_name="Final verification")

    # should be set to true above
    if export_to_excel:
        export_data()


if __name__ == "__main__":
    main(sys.argv[1:])
