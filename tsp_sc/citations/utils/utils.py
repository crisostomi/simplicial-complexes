from tsp_sc.citations.models.my_scnn import MySCNN
from tsp_sc.citations.models.deff_scnn import DeffSCNN
from tsp_sc.citations.models.deff_scnn_mlp import DeffSCNNMLP


def get_model(model_name, model_params):
    if model_name.startswith("my_scnn"):
        return MySCNN(model_params)
    elif model_name.startswith("deff_scnn_mlp"):
        return DeffSCNNMLP(model_params)
    elif model_name.startswith("deff_scnn"):
        return DeffSCNN(model_params)
    else:
        return "Model name does not exist."
