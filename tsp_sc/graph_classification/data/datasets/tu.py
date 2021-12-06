import os
import torch
import pickle
import numpy as np
import random
from tsp_sc.graph_classification.data.tu_utils import (
    load_data,
    S2V_to_PyG,
)
from tsp_sc.common.bodnar_utils import convert_graph_dataset_with_gudhi
from tsp_sc.graph_classification.data.dataset import InMemoryComplexDataset
from sklearn.model_selection import train_test_split
from tsp_sc.common.misc import Phases
from math import ceil


class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(
        self,
        root,
        name,
        max_dim=2,
        num_classes=2,
        degree_as_tag=False,
        init_method="sum",
        seed=0,
        fold=None,
        num_folds=10,
    ):
        self.name = name
        self.degree_as_tag = degree_as_tag
        self.root = root

        self.split_filenames = {
            phase: os.path.join(root, f"split/{phase.value}") for phase in Phases
        }

        super(TUDataset, self).__init__(
            root, max_dim=max_dim, num_classes=num_classes, init_method=init_method,
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.seed = seed

        self.split_indices = (
            self.get_split_indices()
            if fold is None
            else self.get_k_fold_indices(fold, num_folds)
        )

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
        print(f"Computed a total of {len(complexes)} complexes")
        complexes = [complex for complex in complexes if complex.triangles is not None]
        print(f"Only {len(complexes)} have triangles")

        self.create_splits(complexes)

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])

    def get_k_fold_indices(self, fold, num_folds):
        N = self.len()
        indices = np.arange(N)
        width = ceil(N / num_folds)

        start = width * fold
        end = width * (fold + 1)

        val_indices = indices[start:end]
        training_indices = np.concatenate((indices[:start], indices[end:]))

        indices = {Phases.train: training_indices, Phases.val: val_indices}
        return indices

    def create_splits(self, samples):

        indices = np.arange(len(samples))
        random.shuffle(indices)

        y = np.arange(len(samples))
        x_train, x_valtest, y_train, y_valtest = train_test_split(
            indices, y, test_size=0.2
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_valtest, y_valtest, test_size=0.5
        )

        np.save(self.split_filenames[Phases.train], x_train)
        np.save(self.split_filenames[Phases.val], x_val)
        np.save(self.split_filenames[Phases.test], x_test)

    def get_split_indices(self):
        indices = {
            phase: np.load(f"{self.split_filenames[phase]}.npy") for phase in Phases
        }
        return indices
