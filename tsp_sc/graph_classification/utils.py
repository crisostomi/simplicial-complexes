def get_paths(path_params, data_params):
    dataset = data_params["dataset"]
    paths = {k: v for k, v in path_params.items()}
    dataset_const = "DATASET"
    for path_name, path_value in path_params.items():
        if dataset_const in path_value:
            paths[path_name] = path_value.replace(dataset_const, dataset)
    return paths
