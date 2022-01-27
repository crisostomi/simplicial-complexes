import os
import torch
import pickle
import numpy as np

from tsp_sc.graph_classification.data.tu_utils import load_data, get_label_dict
from tsp_sc.common.preproc_utils import convert_graph_dataset_with_gudhi
from tsp_sc.graph_classification.data.in_memory_dataset import InMemoryComplexDataset
from tsp_sc.common.misc import Phases
from tsp_sc.common.utils import eliminate_zeros, sparse_tensor_to_sparse_matrix
from tsp_sc.graph_classification.data.tu_utils import get_fold_indices
import scipy


class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(
        self,
        root,
        name,
        max_dim=2,
        num_classes=2,
        attr_to_consider="tag",
        init_method="sum",
        seed=0,
        fold=0,
    ):
        self.name = name
        self.attr_to_consider = attr_to_consider
        self.root = root
        self.fold = fold
        self.seed = seed

        self.split_filenames = {
            phase: os.path.join(root, f"split/{phase.value}") for phase in Phases
        }
        self.ignore_idxs_path = os.path.join(self.raw_dir, "ignore_idxs.txt")

        super(TUDataset, self).__init__(
            root, max_dim=max_dim, num_classes=num_classes, init_method=init_method,
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.tensors_to_sparse_matrices()

        indices_dir = os.path.join(self.raw_dir, "10fold_idx")
        train_filename = os.path.join(indices_dir, f"train_idx-{fold+1}.txt")
        val_filename = os.path.join(indices_dir, f"test_idx-{fold+1}.txt")

        if os.path.isfile(train_filename) and os.path.isfile(val_filename):
            print(f"Loading indices from existing files, fold: {fold+1}")
            train_idxs = np.loadtxt(train_filename, dtype=int).tolist()
            val_idxs = np.loadtxt(val_filename, dtype=int).tolist()
        else:
            train_idxs, val_idxs = get_fold_indices(self, self.seed, self.fold)

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
        return [f"{self.name}_graph_list_{self.attr_to_consider}.pkl"]

    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data_list = load_data(
            self.raw_dir, self.name, attr_to_consider=self.attr_to_consider
        )
        label_dict = get_label_dict(data_list)
        self._num_classes = len(label_dict)

        with open(self.raw_paths[0], "wb") as handle:
            pickle.dump(data_list, handle)

    def process(self):
        with open(self.raw_paths[0], "rb") as handle:
            graph_list = pickle.load(handle)

        print("Converting the dataset with gudhi...")

        complexes = convert_graph_dataset_with_gudhi(
            graph_list, expansion_dim=self.max_dim, init_method=self._init_method,
        )

        self.ignore_idxs = np.array(
            [ind for ind, complex in enumerate(complexes) if complex.triangles is None],
            dtype=int,
        )

        np.savetxt(self.ignore_idxs_path, self.ignore_idxs, fmt="%d")

        collated_dataset = self.collate(complexes, self.max_dim)

        print(f"Saving dataset..")
        torch.save(collated_dataset, self.processed_paths[0])

    def tensors_to_sparse_matrices(self):
        """
        Converts each sparse tensor to a scipy sparse matrix
        which better supports sparse operations
        """

        for dim in range(self.max_dim + 1):

            for key in self.data[dim].keys:

                value = self.data[dim][key]

                if torch.is_tensor(value) and value.is_sparse:

                    sparse_mat = sparse_tensor_to_sparse_matrix(value)

                    sparse_mat.eliminate_zeros()

                    self.data[dim][key] = sparse_mat
