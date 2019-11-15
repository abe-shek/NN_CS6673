import getopt
import sys

import numpy as np


class Model:
    def __init__(self, n0, n1, n2):
        self.n_0 = n0
        self.n_1 = n1
        self.n_2 = n2
        self.weights_1 = np.array(len(self.n_0), len(self.n_1))
        self.weights_2 = np.array(len(self.n_1), len(self.n_2))
        self.biases_1 = np.array(1, len(self.n_1))
        self.biases_2 = np.array(1, len(self.n_2))
        self.activations_1 = np.array(1, len(self.n_1))
        self.activations_2 = np.array(1, len(self.n_2))
        self.sensitivities_1 = np.array(1, len(self.n_1))
        self.sensitivities_2 = np.array(1, len(self.n_2))
        self.hyper_params = HyperParams


class HyperParams:
    def __init__(self, no_training_steps=800, alpha=0.2, alpha_sch=2, percentage=0.98):
        self.training_steps = no_training_steps
        self.learning_rate = alpha
        self.lr_scheduling_option = alpha_sch  # {0: no_sch, 1: step_sch, 2: per_sch}
        self.lr_perc_decrease = percentage


def train_model(s, t, model):
    for step in range(1, model.hyper_params.training_steps):
        if model.hyper_params.lr_scheduling_option == 0:
            model.hyper_params.learning_rate = model.hyper_params.learning_rate
        elif model.hyper_params.lr_scheduling_option == 1:
            model.hyper_params.learning_rate = model.hyper_params.learning_rate/step
        elif model.hyper_params.lr_scheduling_option == 2:
            model.hyper_params.learning_rate = model.f * model.hyper_params.learning_rate

        x_rand = s[np.random.randint(0,len(s))]
        y = t[s.index(x_rand)]
        activations_0 = np.array(x_rand).reshape(1, 2)

        # forward propagation
        model.activations_1 = transfer_fn(activations_0.dot(model.weights_1) + model.biases_1)
        model.activations_2 = transfer_fn(model.activations_1.dot(model.weights_2) + model.biases_2)

        # backward propagation
        model.sensitivities_2 = (model.activations_2 - np.array(y).reshape(1,1))\
            .dot(trans_fn_first_derivative(model.activations_2))
        model.sensitivities_1 = np.multiply(trans_fn_first_derivative(model.activations_1),
            (model.weights_2.dot(model.sensitivities_2.transpose())).transpose())

        # update weights and biases
        model.weights_1 = model.weights_1 - (model.hyper_params.learning_rate *
                                             (activations_0.transpose().dot(model.sensitivities_1)))
        model.biases_1 = model.biases_1 - (model.hyper_params.learning_rate * model.sensitivities_1)
        model.weights_2 = model.weights_2 - (model.hyper_params.learning_rate *
                                             (model.activations_1.transpose().dot(model.sensitivities_2)))
        model.biases_2 = model.biases_2 - (model.hyper_params.learning_rate * model.sensitivities_2)

    return model

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["steps=","alpha=","asch=","perc="])
    except getopt.GetoptError:
        print("Use - python main.py --steps <num_of_training_steps> --alpha <alpha> "
              "--asch <alpha_scheduling_option> --perc <f_perc_value>")
        sys.exit(2)
    model = Model(2, 4, 1)
    for opt, arg in opts:
        if opt == "--steps":
            model.hyper_params.training_steps = int(arg)
        elif opt == "--alpha":
            model.hyper_params.learning_rate = float(arg)
        elif opt == "--asch":
            model.hyper_params.lr_scheduling_option = int(arg)
        elif opt == "--perc":
            model.hyper_params.lr_perc_decrease = float(arg)

    s = []
    t = []
    model = train_model(s, t, model)
    # validate_nn(model)
    # info(model)
