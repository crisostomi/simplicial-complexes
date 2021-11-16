from tsp_sc.citations.models.my_scnn import MySCNN
from tsp_sc.citations.models.deff_scnn import DeffSCNN
from tsp_sc.citations.models.deff_scnn_mlp import DeffSCNNMLP
from wandb import Config


def fix_dict_in_config(wandb):
    config = dict(wandb.config)
    for k, v in config.copy().items():

        if "." in k:
            first_nested_key = k.split(".")[0]
            remaining_keys = ".".join(k.split(".")[1:])

            if first_nested_key not in config.keys():
                config[first_nested_key] = {}

            if "." in remaining_keys:
                second_nested_key = remaining_keys.split(".")[0]
                third_nested_key = remaining_keys.split(".")[1]

                if second_nested_key not in config[first_nested_key]:
                    config[second_nested_key] = {}

                config[first_nested_key][second_nested_key].update(
                    {third_nested_key: v}
                )
            else:
                config[first_nested_key].update({remaining_keys: v})

            del config[k]

    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v


def get_model(model_name, model_params):
    if model_name.startswith("my_scnn"):
        return MySCNN(model_params)
    elif model_name.startswith("deff_scnn_mlp"):
        return DeffSCNNMLP(model_params)
    elif model_name.startswith("deff_scnn"):
        return DeffSCNN(model_params)
    else:
        return "Model name does not exist."
