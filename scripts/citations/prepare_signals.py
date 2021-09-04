import argparse
from tsp_sc.common.io import load_config
from tsp_sc.citations.utils.citations import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("starting_node")
    parser.add_argument("perc_missing")

    cli_args = parser.parse_args()
    config = load_config(cli_args.config)
    paths = config["paths"]

    starting_node = cli_args.starting_node

    percentage_missing_values = cli_args.perc_missing

    complex_folder_node = os.path.join(paths["complex_folder"], starting_node)
    cochains = np.load(
        os.path.join(complex_folder_node, "cochains.npy"), allow_pickle=True
    )
    simplices = np.load(
        os.path.join(complex_folder_node, "simplices.npy"), allow_pickle=True
    )

    missing_values = build_missing_values(
        simplices, percentage_missing_values=30, max_dim=7
    )

    damaged_dataset = build_damaged_dataset(
        cochains, missing_values, function=np.median
    )

    known_values = build_known_values(missing_values, simplices)

    # Save
    missing_values_path = os.path.join(
        complex_folder_node,
        f"percentage_{percentage_missing_values}_missing_values.npy",
    )
    damaged_input_path = os.path.join(
        complex_folder_node,
        f"percentage_{percentage_missing_values}_input_damaged.npy",
    )

    known_values_path = os.path.join(
        complex_folder_node, f"percentage_{percentage_missing_values}_known_values.npy",
    )

    np.save(missing_values_path, missing_values)
    np.save(damaged_input_path, damaged_dataset)
    np.save(known_values_path, known_values)
