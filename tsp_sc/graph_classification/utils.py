from tsp_sc.graph_classification.models.classification_scnn import ClassificationSCNN
from tsp_sc.graph_classification.data.datasets.tu import TUDataset
import os
from tsp_sc.graph_classification.data.dataset import ComplexDataset

DATASET_NAMES = {
    "IMDBBINARY",
    "IMDBMULTI",
    "NCI1",
    "PROTEINS",
    "REDDITBINARY",
    "REDDITMULTI5K",
}


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


def load_dataset(name, root, max_dim=2, init_method="sum", fold=None) -> ComplexDataset:
    """
    Returns a ComplexDataset with the specified name and initialised with the given params.
    """

    assert name in DATASET_NAMES

    num_classes = {
        "PROTEINS": 2,
        "NCI1": 2,
        "REDDITBINARY": 2,
        "IMDBBINARY": 2,
        "IMDBMULTI": 3,
        "REDDITMULTI5K": 5,
    }

    attr_to_consider = "degree" if name in {"IMDBBINARY", "IMDBMULTI"} else "tag"

    dataset = TUDataset(
        os.path.join(root, name),
        name,
        max_dim=max_dim,
        num_classes=num_classes[name],
        attr_to_consider=attr_to_consider,
        init_method=init_method,
        fold=fold,
    )
    return dataset
