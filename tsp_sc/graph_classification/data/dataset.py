from torch_geometric.data import Dataset
import copy
import re
from abc import ABC

import torch
import os.path as osp
from tsp_sc.common.utils import sparse_slice
from torch_geometric.data import Dataset
from itertools import repeat, product
from tsp_sc.common.simp_complex import SimplicialComplex, Cochain
from tsp_sc.common.utils import block_diagonal
from torch import Tensor
from torch_sparse import cat as sparse_cat


class ComplexDataset(Dataset, ABC):
    """
    Base class for cochain complex datasets.

    This class mirrors
    https://github.com/rusty1s/pytorch_geometric/blob/76d61eaa9fc8702aa25f29dfaa5134a169d0f1f6/torch_geometric/data/dataset.py#L19
    """

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_dim: int = None,
        num_classes: int = None,
        init_method=None,
    ):
        # These have to be initialised before calling the super class.
        self._max_dim = max_dim
        self._num_features = [None for _ in range(max_dim + 1)]
        self._init_method = init_method

        super(ComplexDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self._num_classes = num_classes
        self.train_ids = None
        self.val_ids = None
        self.test_ids = None

    @property
    def max_dim(self):
        return self._max_dim

    @max_dim.setter
    def max_dim(self, value):
        self._max_dim = value

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_dir(self):
        return osp.join(self.root, f"complex_dim{self.max_dim}_{self._init_method}")

    def num_features_in_dim(self, dim):
        """
        :param dim:
        :return:
        """
        if dim > self.max_dim:
            raise ValueError(
                "`dim` {} larger than max allowed dimension {}.".format(
                    dim, self.max_dim
                )
            )
        if self._num_features[dim] is None:
            self._look_up_num_features()
        return self._num_features[dim]

    def _look_up_num_features(self):
        """
        Set num features for each dimension. All complexes must have
        the same number of features for the same dimension.
        :return:
        """
        for complex in self:
            for dim in range(complex.dimension + 1):
                if self._num_features[dim] is None:
                    self._num_features[dim] = complex.cochains[dim].num_features
                else:
                    assert self._num_features[dim] == complex.cochains[dim].num_features

    def get_idx_split(self):
        idx_split = {
            "train": self.train_ids,
            "valid": self.val_ids,
            "test": self.test_ids,
        }
        return idx_split


class InMemoryComplexDataset(ComplexDataset):
    """Wrapper around ComplexDataset with functionality such as batching and storing the dataset.

    This class mirrors
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py
    """

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_dim: int = None,
        num_classes: int = None,
        init_method=None,
    ):
        super(InMemoryComplexDataset, self).__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            max_dim,
            num_classes,
            init_method=init_method,
        )
        self.data, self.slices = None, None
        self.__data_list__ = None

    def len(self):
        for dim in range(self.max_dim + 1):
            for item in self.slices[dim].values():
                return len(item) - 1
        return 0

    def get(self, idx):

        if hasattr(self, "__data_list__"):
            if self.__data_list__ is None:
                self.__data_list__ = self.len() * [None]
            else:
                data = self.__data_list__[idx]
                if data is not None:
                    return copy.copy(data)

        retrieved = [self._get_cochain(dim, idx) for dim in range(0, self.max_dim + 1)]
        cochains = [r[0] for r in retrieved if not r[1]]

        targets = self.data["labels"]
        start, end = idx, idx + 1
        if torch.is_tensor(targets):
            s = list(repeat(slice(None), targets.dim()))
            cat_dim = 0
            s[cat_dim] = slice(start, end)
        else:
            # TODO: come up with a better method to handle this
            assert targets[start] is None
            s = start

        target = targets[s]

        dim = self.data["dims"][idx].item()
        assert dim == len(cochains) - 1
        data = SimplicialComplex(*cochains, y=target)

        if hasattr(self, "__data_list__"):
            self.__data_list__[idx] = copy.copy(data)

        return data

    def _get_cochain(self, dim, idx) -> (Cochain, bool):

        if dim < 0 or dim > self.max_dim:
            raise ValueError(
                f"The current dataset does not have cochains at dimension {dim}."
            )

        cochain_data = self.data[dim]
        cochain_slices = self.slices[dim]

        data = Cochain(dim)

        if cochain_data.__num_simplices__[idx] is not None:
            data.num_simplices = cochain_data.__num_simplices__[idx]
        if cochain_data.__num_simplices_up__[idx] is not None:
            data.num_simplices_up = cochain_data.__num_simplices_up__[idx]
        if cochain_data.__num_simplices_down__[idx] is not None:
            data.num_simplices_down = cochain_data.__num_simplices_down__[idx]
        elif dim == 0:
            data.num_simplices_down = None

        for key in cochain_data.keys:
            item, slices = cochain_data[key], cochain_slices[key]
            data[key] = None
            if torch.is_tensor(item) and item.is_sparse:
                start, end = slices[idx], slices[idx + 1]
                data[key] = sparse_slice(item, start, end)
            else:
                start, end = slices[idx].item(), slices[idx + 1].item()
                if start != end:
                    if torch.is_tensor(item):
                        s = list(repeat(slice(None), item.dim()))
                        cat_dim = cochain_data.__cat_dim__(key, item)
                        if cat_dim is None:
                            cat_dim = 0
                        s[cat_dim] = slice(start, end)
                    elif start + 1 == end:
                        s = slices[start]
                    else:
                        s = slice(start, end)
                    data[key] = item[s]

        empty = data.num_simplices is None

        return data, empty

    @staticmethod
    def collate(data_list, max_dim):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`InMemoryComplexDataset`."""

        sparse_keys = [
            "boundary",
            "laplacian",
            "solenoidal",
            "irrotational",
            "coboundary",
        ]

        def init_keys(dim, keys, sparse_keys):
            cochain = Cochain(dim)
            for key in keys[dim]:
                cochain[key] = []
            cochain.__num_simplices__ = []
            cochain.__num_simplices_up__ = []
            cochain.__num_simplices_down__ = []
            slc = {
                key: [torch.tensor([0, 0])] if key in sparse_keys else [torch.tensor(0)]
                for key in keys[dim]
            }
            return cochain, slc

        def collect_keys(data_list, max_dim):
            keys = {dim: set() for dim in range(0, max_dim + 1)}
            for complex in data_list:
                for dim in keys:
                    if dim not in complex.cochains:
                        continue
                    cochain = complex.cochains[dim]
                    keys[dim] |= set(cochain.keys)
            return keys

        keys = collect_keys(data_list, max_dim)
        types = {}
        cat_dims = {}
        tensor_dims = {}
        slices = {}
        data = {"labels": [], "dims": []}

        for dim in range(0, max_dim + 1):
            data[dim], slices[dim] = init_keys(dim, keys, sparse_keys)

        for complex in data_list:

            # Collect cochain-wise items
            for dim in range(0, max_dim + 1):

                # Get cochain, if present
                cochain = None
                if dim in complex.cochains:
                    cochain = complex.cochains[dim]

                # Iterate on keys
                for key in keys[dim]:
                    if (
                        cochain is not None
                        and hasattr(cochain, key)
                        and cochain[key] is not None
                    ):
                        data[dim][key].append(cochain[key])
                        if isinstance(cochain[key], Tensor) and cochain[key].dim() > 0:
                            cat_dim = cochain.__cat_dim__(key, cochain[key])
                            cat_dim = 0 if cat_dim is None else cat_dim
                            if cochain[key].is_sparse:
                                size = torch.tensor(cochain[key].size())[
                                    torch.tensor(cat_dim)
                                ]
                            else:
                                size = cochain[key].size(cat_dim)
                            s = slices[dim][key][-1] + size
                            if key not in cat_dims:
                                cat_dims[key] = cat_dim
                            else:
                                assert cat_dim == cat_dims[key]
                            if key not in tensor_dims:
                                tensor_dims[key] = cochain[key].dim()
                            else:
                                assert cochain[key].dim() == tensor_dims[key]
                        else:
                            s = slices[dim][key][-1] + 1
                        if key not in types:
                            types[key] = type(cochain[key])
                        else:
                            assert type(cochain[key]) is types[key]
                    else:
                        s = slices[dim][key][-1]
                    slices[dim][key].append(s)

                # Handle non-keys
                # TODO: could they be considered as keys as well?
                num = None
                num_up = None
                num_down = None
                if cochain is not None:
                    if hasattr(cochain, "__num_simplices__"):
                        num = cochain.__num_simplices__
                    if hasattr(cochain, "__num_simplices_up__"):
                        num_up = cochain.__num_simplices_up__
                    if hasattr(cochain, "__num_simplices_down__"):
                        num_down = cochain.__num_simplices_down__
                data[dim].__num_simplices__.append(num)
                data[dim].__num_simplices_up__.append(num_up)
                data[dim].__num_simplices_down__.append(num_down)

            # Collect complex-wise label(s) and dims
            if not hasattr(complex, "y"):
                complex.y = None
            if isinstance(complex.y, Tensor):
                assert complex.y.size(0) == 1
            data["labels"].append(complex.y)
            data["dims"].append(complex.dimension)

        # Pack lists into tensors

        # Cochains
        for dim in range(0, max_dim + 1):
            for key in keys[dim]:
                if types[key] is Tensor and len(data_list) > 1:
                    if tensor_dims[key] > 0:
                        cat_dim = cat_dims[key]
                        if data[dim][key][0].is_sparse:
                            data[dim][key] = block_diagonal(*data[dim][key])
                        else:
                            data[dim][key] = torch.cat(data[dim][key], dim=cat_dim)
                    else:
                        data[dim][key] = torch.stack(data[dim][key])
                elif types[key] is Tensor:  # Don't duplicate attributes...
                    data[dim][key] = data[dim][key][0]
                elif types[key] is int or types[key] is float:
                    data[dim][key] = torch.tensor(data[dim][key])
                # tuple as index
                if key in sparse_keys:
                    tensors = [torch.tensor(pair) for pair in slices[dim][key]]
                    slices[dim][key] = torch.stack(tensors)
                else:
                    slices[dim][key] = torch.tensor(slices[dim][key], dtype=torch.long)

        # Labels and dims
        item = data["labels"][0]
        if isinstance(item, Tensor) and len(data_list) > 1:
            if item.dim() > 0:
                cat_dim = 0
                data["labels"] = torch.cat(data["labels"], dim=cat_dim)
            else:
                data["labels"] = torch.stack(data["labels"])
        elif isinstance(item, Tensor):
            data["labels"] = data["labels"][0]
        elif isinstance(item, int) or isinstance(item, float):
            data["labels"] = torch.tensor(data["labels"])
        data["dims"] = torch.tensor(data["dims"])

        return data, slices

    def copy(self, idx=None):
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in idx]
        dataset = copy.copy(self)
        dataset.__indices__ = None
        dataset.__data_list__ = data_list
        dataset.data, dataset.slices = self.collate(data_list)

        return dataset

    def get_split(self, split):
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Unknown split {split}.")
        idx = self.get_idx_split()[split]
        if idx is None:
            raise AssertionError("No split information found.")
        if self.__indices__ is not None:
            raise AssertionError(
                "Cannot get the split for a subset of the original dataset."
            )
        return self[idx]
