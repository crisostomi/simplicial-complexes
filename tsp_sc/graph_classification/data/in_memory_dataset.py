import copy
import torch
from tsp_sc.common.utils import sparse_slice
from itertools import repeat
from tsp_sc.common.simp_complex import SimplicialComplex, Cochain
from tsp_sc.common.utils import block_diagonal
from tsp_sc.common.constants import SPARSE_KEYS
from torch import Tensor
import scipy
from tsp_sc.graph_classification.data.dataset import ComplexDataset
from tsp_sc.common.utils import eliminate_zeros


class InMemoryComplexDataset(ComplexDataset):
    """Wrapper around ComplexDataset with functionality such as batching and storing the dataset.

    This class mirrors
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py
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
        super(InMemoryComplexDataset, self).__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            max_dim=max_dim,
            num_classes=num_classes,
            init_method=init_method,
        )

        # data contains labels, dims and the cochains for all the dimensions
        self.data = None

        # slices contains for each dimension the slices for each attribute
        self.slices = None

        # list of Complexes
        self.__data_list__ = None

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

    def len(self):
        """
        :return: number of complexes in the dataset
        """
        num_complexes = len(next(iter(self.slices[0].values())))
        return num_complexes

    def get(self, idx):
        """
        Returns the complex at index idx
        """

        if hasattr(self, "__data_list__"):
            if self.__data_list__ is None:
                self.__data_list__ = self.len() * [None]
            else:
                data = self.__data_list__[idx]
                if data is not None:
                    return copy.copy(data)

        retrieved_cochains = [
            self._get_cochain(dim, idx) for dim in range(0, self.max_dim + 1)
        ]

        # only keep non empty cochains
        cochains = [cochain for cochain, is_empty in retrieved_cochains if not is_empty]

        target = self._get_complex_target(idx)

        dim = self.data["dims"][idx].item()
        assert dim == len(cochains) - 1

        data = SimplicialComplex(*cochains, y=target, dimension=dim)

        if hasattr(self, "__data_list__"):
            self.__data_list__[idx] = copy.copy(data)

        return data

    def _get_cochain(self, dim, idx) -> (Cochain, bool):
        f"""
        Returns the cochain of dimension {dim} of complex at index {idx} 
        """

        if dim < 0 or dim > self.max_dim:
            raise ValueError(
                f"The current dataset does not have cochains at dimension {dim}."
            )

        cochain_data = self.data[dim]
        cochain_slices = self.slices[dim]

        retrieved_cochain = Cochain(dim)

        retrieved_cochain.set_num_simplices_from_cochain(cochain_data, idx, dim)

        for key in cochain_data.keys:
            item, slices = cochain_data[key], cochain_slices[key]
            retrieved_cochain[key] = self.retrieve_slice(
                key, item, idx, slices, cochain_data
            )

        cochain_is_empty = retrieved_cochain.num_simplices is None

        return retrieved_cochain, cochain_is_empty

    @staticmethod
    def retrieve_slice(key, item, idx, slices, cochain_data):

        if scipy.sparse.issparse(item):
            start, end = slices[idx], slices[idx + 1]
            return sparse_slice(item, start, end)

        else:
            start, end = slices[idx].item(), slices[idx + 1].item()

            if start == end:
                return None

            s = list(repeat(slice(None), item.dim()))

            cat_dim = cochain_data.__cat_dim__(key, item)
            if cat_dim is None:
                cat_dim = 0

            s[cat_dim] = slice(start, end)
            return item[s]

    def _get_complex_target(self, idx):
        f"""
        Get target for complex at index {idx}
        """
        targets = self.data["labels"]
        start, end = idx, idx + 1

        if torch.is_tensor(targets):
            s = list(repeat(slice(None), targets.dim()))
            cat_dim = 0
            s[cat_dim] = slice(start, end)
        else:
            assert targets[start] is None
            s = start

        target = targets[s]
        return target

    @staticmethod
    def collate(data_list, max_dim):
        f"""
        Collates a python list of data objects to the internal storage
        format of :class:`InMemoryComplexDataset`. Used in processing.
        :param data_list: list of complexes
        :param max_dim: maximum dimension of the complexes
        """

        keys = InMemoryComplexDataset.collect_keys(data_list, max_dim)

        types = {}
        cat_dims = {}
        tensor_dims = {}
        slices = {}
        data = {"labels": [], "dims": []}

        for dim in range(0, max_dim + 1):
            data[dim] = InMemoryComplexDataset.init_cochain(dim, keys)
            slices[dim] = Cochain.init_cochain_slices(keys[dim])

        InMemoryComplexDataset.collect_cochain_wise_items(
            data, keys, data_list, max_dim, tensor_dims, types, slices, cat_dims
        )

        InMemoryComplexDataset.pack_lists_into_tensors(
            data_list, keys, max_dim, slices, tensor_dims, types, data, cat_dims
        )

        InMemoryComplexDataset.remove_zero_entries(data)

        return data, slices

    @staticmethod
    def collect_keys(data_list, max_dim):
        """
        Collects the keys for the cochains at each dimension
        """
        keys = {dim: set() for dim in range(0, max_dim + 1)}
        for complex in data_list:
            for dim in keys:
                if dim not in complex.cochains:
                    continue
                cochain = complex.cochains[dim]
                keys[dim] |= set(cochain.keys)
        return keys

    @staticmethod
    def init_cochain(dim, keys):
        f"""
        Initializes a cochain of dimension {dim} 
        """
        cochain = Cochain(dim)

        for key in keys[dim]:
            cochain[key] = []

        cochain.__num_simplices__ = []
        cochain.__num_simplices_up__ = []
        cochain.__num_simplices_down__ = []

        return cochain

    @staticmethod
    def collect_cochain_wise_items(
        data, keys, data_list, max_dim, tensor_dims, types, slices, cat_dims
    ):
        """
        Iterates over each complex and collects the items for its cochains at each dimension
        :param data:
        :param keys:
        :param data_list:
        :param max_dim:
        :param tensor_dims:
        :param types:
        :param slices:
        :param cat_dims:
        :return:
        """
        for complex in data_list:

            for dim in range(0, max_dim + 1):

                cochain = complex.cochains[dim] if dim in complex.cochains else None

                InMemoryComplexDataset.handle_keys(
                    keys, cochain, data, slices, cat_dims, dim, types, tensor_dims
                )

                InMemoryComplexDataset.handle_non_keys(cochain, data, dim)

            # Collect complex-wise label(s) and dims
            if not hasattr(complex, "y"):
                complex.y = None
            if isinstance(complex.y, Tensor):
                assert complex.y.size(0) == 1

            data["labels"].append(complex.y)
            data["dims"].append(complex.dimension)

    @staticmethod
    def handle_keys(keys, cochain, data, slices, cat_dims, dim, types, tensor_dims):
        """
        """
        for key in keys[dim]:
            if (
                cochain is not None
                and hasattr(cochain, key)
                and cochain[key] is not None
            ):
                data[dim][key].append(cochain[key])

                if torch.is_tensor(cochain[key]) and cochain[key].dim() > 0:

                    tensor = cochain[key]

                    cat_dim = cochain.__cat_dim__(key, tensor)
                    cat_dim = 0 if cat_dim is None else cat_dim

                    size = Cochain.get_tensor_size(tensor, key, cat_dim)
                    s = slices[dim][key][-1] + size

                    InMemoryComplexDataset.handle_cat_dim(key, cat_dims, cat_dim)
                    InMemoryComplexDataset.handle_tensor_dim(key, cochain, tensor_dims)

                else:
                    s = slices[dim][key][-1] + 1

                InMemoryComplexDataset.handle_types(key, types, cochain)

            else:
                s = slices[dim][key][-1]

            slices[dim][key].append(s)

    @staticmethod
    def handle_non_keys(cochain, data, dim):
        num, num_up, num_down = None, None, None

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

    @staticmethod
    def handle_cat_dim(key, cat_dims, cat_dim):
        if key not in cat_dims:
            cat_dims[key] = cat_dim
        else:
            assert cat_dim == cat_dims[key]

    @staticmethod
    def handle_tensor_dim(key, cochain, tensor_dims):
        if key not in tensor_dims:
            tensor_dims[key] = cochain[key].dim()
        else:
            assert cochain[key].dim() == tensor_dims[key]

    @staticmethod
    def handle_types(key, types, cochain):
        if key not in types:
            types[key] = type(cochain[key])
        else:
            assert type(cochain[key]) is types[key]

    @staticmethod
    def pack_lists_into_tensors(
        data_list, keys, max_dim, slices, tensor_dims, types, data, cat_dims
    ):

        for dim in range(0, max_dim + 1):

            for key in keys[dim]:

                if types[key] is Tensor and len(data_list) > 1:
                    InMemoryComplexDataset.pack_tensors(
                        tensor_dims, cat_dims, key, data, dim
                    )

                elif types[key] is Tensor:  # single element
                    data[dim][key] = data[dim][key][0]

                elif types[key] in (int, float):
                    data[dim][key] = torch.tensor(data[dim][key])

                else:
                    raise NotImplementedError(f"Unsupported type {types[key]}")

                InMemoryComplexDataset.pack_slices(key, slices, dim)

        InMemoryComplexDataset.handle_labels(data, data_list)
        data["dims"] = torch.tensor(data["dims"])

    @staticmethod
    def pack_tensors(tensor_dims, cat_dims, key, data, dim):
        if tensor_dims[key] > 0:
            cat_dim = cat_dims[key]
            if data[dim][key][0].is_sparse:
                data[dim][key] = block_diagonal(*data[dim][key])
            else:
                data[dim][key] = torch.cat(data[dim][key], dim=cat_dim)
        else:
            data[dim][key] = torch.stack(data[dim][key])

    @staticmethod
    def pack_slices(key, slices, dim):
        if key in SPARSE_KEYS:
            tensors = [pair for pair in slices[dim][key]]
            slices[dim][key] = torch.stack(tensors)
        else:
            slices[dim][key] = torch.tensor(slices[dim][key], dtype=torch.long)

    @staticmethod
    def handle_labels(data, data_list):
        item = data["labels"][0]

        if torch.is_tensor(item):
            if len(data_list) > 1:
                if item.dim() > 0:
                    cat_dim = 0
                    data["labels"] = torch.cat(data["labels"], dim=cat_dim)
                else:
                    data["labels"] = torch.stack(data["labels"])
            else:
                data["labels"] = data["labels"][0]

        elif isinstance(item, int) or isinstance(item, float):
            data["labels"] = torch.tensor(data["labels"])

        else:
            raise ValueError(f"Unsupported type {type(item)}")

    @staticmethod
    def remove_zero_entries(data):

        for key, value in data.items():
            if key in {0, 1, 2}:
                cochain = value
                for cochain_key in cochain.keys:
                    cochain_value = cochain[cochain_key]
                    if torch.is_tensor(cochain_value) and cochain_value.is_sparse:
                        cochain[cochain_key] = eliminate_zeros(cochain_value)
