import os
import torch
import pickle
import numpy as np
from tsp_sc.graph_classification.data.tu_utils import (
    load_data,
    S2V_to_PyG,
)
from tsp_sc.common.bodnar_utils import convert_graph_dataset_with_gudhi
from tsp_sc.graph_classification.data.dataset import InMemoryComplexDataset
from tsp_sc.common.misc import Phases


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
        fold=0,
        num_folds=10,
    ):
        self.name = name
        self.degree_as_tag = degree_as_tag
        self.root = root

        self.split_filenames = {
            phase: os.path.join(root, f"split/{phase.value}") for phase in Phases
        }
        self.ignore_idxs_path = os.path.join(self.raw_dir, "ignore_idxs.txt")

        super(TUDataset, self).__init__(
            root, max_dim=max_dim, num_classes=num_classes, init_method=init_method,
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.seed = seed

        indices_dir = os.path.join(self.raw_dir, "10fold_idx")
        train_filename = os.path.join(indices_dir, f"train_idx-{fold+1}.txt")
        val_filename = os.path.join(indices_dir, f"test_idx-{fold+1}.txt")
        assert os.path.isfile(train_filename) and os.path.isfile(val_filename)
        train_idxs = np.loadtxt(train_filename, dtype=int).tolist()
        val_idxs = np.loadtxt(val_filename, dtype=int).tolist()

        self.ignore_idxs = np.loadtxt(self.ignore_idxs_path, dtype=int)
        self.split_indices = {}
        self.split_indices[Phases.train] = [
            idx for idx in train_idxs if idx not in self.ignore_idxs
        ]
        self.split_indices[Phases.val] = [
            idx for idx in val_idxs if idx not in self.ignore_idxs
        ]

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

        self.ignore_idxs = np.array(
            [ind for ind, complex in enumerate(complexes) if complex.triangles is None],
            dtype=int,
        )

        np.savetxt(self.ignore_idxs_path, self.ignore_idxs, fmt="%d")

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
