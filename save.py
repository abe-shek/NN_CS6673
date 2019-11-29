import pandas as pd

sheets = {}


def update_sheet(writer, sheet_name, sheet_obj):
    df_obj = {
        'Model architecture': sheet_obj['model_arch_list'],
        'Model weights': sheet_obj['model_weight_list'],
        'Model biases': sheet_obj['model_bias_list'],
        '# Training epochs': sheet_obj['total_epochs_req_list'],
        'Learning Rate': sheet_obj['learning_rate_list'],
        'Zeta': sheet_obj['zeta_list'],
        'X0': sheet_obj['x0_list'],
        'Cost Function': sheet_obj['cost_fn_list'],
        'Last epoch error': sheet_obj['last_epoch_error_list'],
        'Did converge?': sheet_obj['converged_list'],
    }
    df = pd.DataFrame(df_obj)
    df.to_excel(writer, sheet_name=sheet_name, index=False)


def save_data(sheet_name, model):
    try:
        sheet_obj = sheets[sheet_name]
    except KeyError:
        sheet_obj = {}
    if not sheet_obj:
        sheet_obj = {'model_arch_list': [], 'model_weight_list': [], 'model_bias_list': [],
                     'total_epochs_req_list': [], 'learning_rate_list': [], 'zeta_list': [],
                     'x0_list': [], 'cost_fn_list': [], 'last_epoch_error_list': [],
                     'converged_list': []}
    sheet_obj['model_arch_list'].append("[ " + str(model.n_0) + ", " + str(model.n_1) +
                                        ", " + str(model.n_2) + "]")
    sheet_obj['model_weight_list'].append("Weights1 = " + str(model.weights_1) +
                                          "\nWeights2 = " + str(model.weights_2))
    sheet_obj['model_bias_list'].append("Biases1 = " + str(model.biases_1) +
                                        "\nBiases2 = " + str(model.biases_2))
    sheet_obj['total_epochs_req_list'].append(model.model_info.total_epochs_req)
    sheet_obj['learning_rate_list'].append(model.hyper_params.learning_rate)
    sheet_obj['zeta_list'].append(model.hyper_params.zeta)
    sheet_obj['x0_list'].append(model.hyper_params.x0)
    sheet_obj['cost_fn_list'].append("Quadratic" if model.hyper_params.cost_fn == 0
                                     else "Cross-Entropy")
    sheet_obj['last_epoch_error_list'].append(model.model_info.last_epoch_error)
    sheet_obj['converged_list'].append("Yes" if model.model_info.converged else "No")
    sheets[sheet_name] = sheet_obj


def export_data():
    print("Starting export")
    writer = pd.ExcelWriter('Results.xlsx', engine='xlsxwriter')
    if not writer:
        print("Error while opening writer. Exiting.")
        return
    for sheet_name in sheets:
        sheet_obj = sheets[sheet_name]
        if not sheet_obj:
            print("Skipping sheet - %s " % sheet_name)
            continue
        print("Updating sheet - %s " % sheet_name)
        update_sheet(writer, sheet_name, sheet_obj)
    writer.save()
    print("Data exported")
