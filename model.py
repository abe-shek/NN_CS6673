import numpy as np


class Model:
    def __init__(self, n0, n1, n2):
        self.n_0 = n0
        self.n_1 = n1
        self.n_2 = n2

        self.reinitialize_weights = True
        self.weights_1 = np.zeros((self.n_0, self.n_1), dtype=int)
        self.weights_2 = np.zeros((self.n_1, self.n_2), dtype=int)
        self.biases_1 = np.zeros((1, self.n_1), dtype=int)
        self.biases_2 = np.zeros((1, self.n_2), dtype=int)

        self.activations_1 = np.zeros((1, self.n_1), dtype=int)
        self.activations_2 = np.zeros((1, self.n_2), dtype=int)
        self.sensitivities_1 = np.zeros((1, self.n_1), dtype=int)
        self.sensitivities_2 = np.zeros((1, self.n_2), dtype=int)

        self.hyper_params = HyperParams()

        self.model_info = Info()


class HyperParams:
    def __init__(self, no_training_steps=800, alpha_sch=2, percentage=0.98):
        self.alpha_list = [0.1, 0.2, 0.3]
        self.zeta_list = [0.5, 1, 1.5]
        self.x0_list = [0.5, 1, 1.5]
        self.max_epochs = 700  # empirically chosen
        self.tolerance = 0.05
        self.training_steps = no_training_steps
        self.learning_rate = self.alpha_list[1]
        self.lr_scheduling_option = alpha_sch  # {0: no_sch, 1: step_sch, 2: per_sch}
        self.lr_perc_decrease = percentage
        self.zeta = self.zeta_list[1]
        self.x0 = self.x0_list[1]
        self.cost_fn = 0  # {0: quadratic, 1: cross-entropy}


class Info:
    def __init__(self, total_epochs=0, last_epoch_error=0.0, convergence=False):
        self.total_epochs_req = total_epochs
        self.last_epoch_error = last_epoch_error
        self.converged = convergence
