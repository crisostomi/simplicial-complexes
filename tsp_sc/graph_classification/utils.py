from tsp_sc.graph_classification.models.classification_scnn import ClassificationSCNN
from tsp_sc.graph_classification.data.datasets.tu import TUDataset
import os
from tsp_sc.graph_classification.data.dataset import ComplexDataset


def get_paths(path_params, data_params):
    dataset = data_params["dataset"]
    paths = {k: v for k, v in path_params.items()}
    dataset_const = "DATASET"
    for path_name, path_value in path_params.items():
        if dataset_const in path_value:
            paths[path_name] = path_value.replace(dataset_const, dataset)
    return paths


def get_model(model_name, model_params):
    if model_name.startswith("classification_scnn"):
        return ClassificationSCNN(model_params)
    else:
        return "Model name does not exist."


def load_dataset(
    name, root, max_dim=2, init_method="sum", fold=None, n_jobs=2, **kwargs
) -> ComplexDataset:
    """Returns a ComplexDataset with the specified name and initialised with the given params."""
    if name in [
        "PROTEINS",
        "IMDBBINARY",
        "IMDBMULTI",
        "MUTAG",
        "NCI1",
        "NCI109",
        "PTC",
        "REDDITBINARY",
        "REDDITMULTI5K",
        "COLLAB",
    ]:
        dataset = TUDataset(
            os.path.join(root, name),
            name,
            max_dim=max_dim,
            num_classes=2,
            degree_as_tag=False,
            init_method=init_method,
            fold=fold,
        )
    else:
        raise NotImplementedError(f"Dataset {name} not supported.")
    return dataset
