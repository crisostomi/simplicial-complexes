import os
import torch
import pickle
import numpy as np

from tsp_sc.graph_classification.data.tu_utils import (
    load_data,
    S2V_to_PyG,
    get_fold_indices,
)
from tsp_sc.common.bodnar_utils import convert_graph_dataset_with_gudhi
from tsp_sc.graph_classification.data.dataset import InMemoryComplexDataset


class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(
        self,
        root,
        name,
        max_dim=2,
        num_classes=2,
        degree_as_tag=False,
        fold=0,
        init_method="sum",
        seed=0,
    ):
        self.name = name
        self.degree_as_tag = degree_as_tag

        super(TUDataset, self).__init__(
            root, max_dim=max_dim, num_classes=num_classes, init_method=init_method,
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.fold = fold
        self.seed = seed
        train_filename = os.path.join(
            self.raw_dir, "10fold_idx", "train_idx-{}.txt".format(fold + 1)
        )
        test_filename = os.path.join(
            self.raw_dir, "10fold_idx", "test_idx-{}.txt".format(fold + 1)
        )

        print(len(self.data))

    @property
    def processed_dir(self):
        directory = super(TUDataset, self).processed_dir
        return directory

    @property
    def processed_file_names(self):
        return ["{}_complex_list.pt".format(self.name)]

    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG.
        return [
            "{}_graph_list_degree_as_tag_{}.pkl".format(self.name, self.degree_as_tag)
        ]

    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data, num_classes = load_data(self.raw_dir, self.name, self.degree_as_tag)
        self._num_classes = num_classes
        print("Converting graph data into PyG format...")
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(self.raw_paths[0], "wb") as handle:
            pickle.dump(graph_list, handle)

    def process(self):
        with open(self.raw_paths[0], "rb") as handle:
            graph_list = pickle.load(handle)

        print("Converting the dataset with gudhi...")
        complexes, _, _ = convert_graph_dataset_with_gudhi(
            graph_list, expansion_dim=self.max_dim, init_method=self._init_method,
        )

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
