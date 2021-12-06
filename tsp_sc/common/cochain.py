from scipy.sparse import coo_matrix
from tsp_sc.common.simplices import normalize_laplacian
from scipy.sparse import linalg
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from tsp_sc.common.utils import block_diagonal
import logging
import copy
from timeit import default_timer as timer


class Cochain:
    def __init__(
        self,
        dim,
        signal=None,
        y=None,
        complex_dim=None,
        boundary=None,
        coboundary=None,
        **kwargs,
    ):

        # dim k: boundary = B_{k+1} coboundary = B_k^T

        self.__dim__ = dim
        self.complex_dim = complex_dim
        self.boundary = boundary
        self.coboundary = coboundary
        self.__signal = signal
        self.y = y

        for key, item in kwargs.items():
            if key == "num_simplices":
                self.__num_simplices__ = item
            elif key == "num_simplices_down":
                self.num_simplices_down = item
            elif key == "num_simplices_up":
                self.num_simplices_up = item
            else:
                self[key] = item

        if self.boundary is not None or self.coboundary is not None:
            self.laplacian = self.build_laplacian()

            self.solenoidal = self.get_solenoidal_component()
            self.irrotational = self.get_irrotational_component()

            self.normalize_components()

            self.represent_as_sparse_matrices()
            assert (
                self.solenoidal is None or self.laplacian.shape == self.solenoidal.shape
            )
            assert (
                self.irrotational is None
                or self.laplacian.shape == self.irrotational.shape
            )

            assert self.laplacian.shape[0] == self.num_simplices

    def get_irrotational_component(self):
        if self.dim == self.complex_dim:
            return None

        Btk_upper = self.boundary.transpose()
        Bk_upper = self.boundary

        BBt = Bk_upper @ Btk_upper
        irr = coo_matrix(BBt)

        return irr

    def get_solenoidal_component(self):
        if self.dim == 0:
            return None

        Btk = self.coboundary
        Bk = self.coboundary.T

        BtB = Btk @ Bk
        sol = coo_matrix(BtB)

        return sol

    def normalize_components(self):
        if self.laplacian.shape == (1, 1):
            return

        lap_largest_eigenvalue = linalg.eigsh(
            self.laplacian, k=1, which="LM", return_eigenvectors=False,
        )[0]

        if self.dim != 0:
            self.solenoidal = normalize_laplacian(
                self.solenoidal, lap_largest_eigenvalue, half_interval=True,
            )
        if self.dim != self.complex_dim:
            self.irrotational = normalize_laplacian(
                self.irrotational, lap_largest_eigenvalue, half_interval=True,
            )
        self.laplacian = normalize_laplacian(
            self.laplacian, lap_largest_eigenvalue, half_interval=True,
        )

    def build_laplacian(self):
        # upper Laplacian B_{k+1} B_{k}^T
        # lower Laplacian B_{k}^T B_k

        if 0 < self.dim < self.complex_dim:
            upper = self.boundary @ self.boundary.T
            lower = self.coboundary @ self.coboundary.T
            return coo_matrix(lower + upper)

        elif self.dim == 0:
            upper = self.boundary @ self.boundary.T
            return coo_matrix(upper)

        else:
            lower = self.coboundary @ self.coboundary.T
            return coo_matrix(lower)

    def represent_as_sparse_matrices(self):
        self.boundary = self.to_torch_sparse_tensor(self.boundary)
        self.coboundary = self.to_torch_sparse_tensor(self.coboundary)
        self.laplacian = self.to_torch_sparse_tensor(self.laplacian)
        self.irrotational = self.to_torch_sparse_tensor(self.irrotational)
        self.solenoidal = self.to_torch_sparse_tensor(self.solenoidal)

    @staticmethod
    def to_torch_sparse_tensor(sparse_matrix):
        if sparse_matrix is None:
            return None
        row = torch.Tensor(sparse_matrix.row)
        col = torch.Tensor(sparse_matrix.col)
        indices = torch.stack((row, col), dim=0)
        data = torch.Tensor(sparse_matrix.data)
        return torch.sparse_coo_tensor(
            indices=indices, values=data, size=sparse_matrix.shape
        )

    @property
    def num_features(self):
        """Returns the number of features per cell in the cochain."""
        if self.signal is None:
            return 0
        return 1 if self.signal.dim() == 1 else self.signal.size(1)

    @property
    def dim(self):
        """Returns the dimension of the cells in this cochain.

        This field should not have a setter. The dimension of a cochain cannot be changed.
        """
        return self.__dim__

    @property
    def signal(self):
        """Returns the vector values (features) associated with the simplices."""
        return self.__signal

    @signal.setter
    def signal(self, new_signal):
        """Sets the vector values (features) associated with the cells."""
        if new_signal is None:
            logging.warning("Cochain features were set to None. ")
        else:
            assert self.num_simplices == len(new_signal)
        self.__signal = new_signal

    @property
    def keys(self):
        """Returns all names of cochain attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __getitem__(self, key):
        """Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def __contains__(self, key):
        """Returns :obj:`True`, if the attribute :obj:`key` is present in the data."""
        return key in self.keys

    def __cat_dim__(self, key, value):
        """
        Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.
        """
        if key in ["laplacian", "boundary", "coboundary", "solenoidal", "irrotational"]:
            return (0, 1)
        else:
            return 0

    def __inc__(self, key, value):
        """
        Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.
        """
        if key in "boundary":
            row_inc = self.num_simplices if self.num_simplices is not None else 0
            col_inc = self.num_simplices_up if self.num_simplices_up is not None else 0
            inc = [row_inc, col_inc]
        elif key == "coboundary":
            row_inc = self.num_simplices_up if self.num_simplices_up is not None else 0
            col_inc = self.num_simplices if self.num_simplices is not None else 0
            inc = [row_inc, col_inc]
        elif key in ["laplacian", "solenoidal", "irrotational"]:
            row_col_inc = self.num_simplices if self.num_simplices is not None else 0
            inc = [row_col_inc, row_col_inc]
        else:
            inc = 0

        return inc

    def __call__(self, *keys):
        """
        Iterates over all attributes :obj:`*keys` in the cochain, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes.
        """
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_simplices(self):
        """Returns the number of simplices in the cochain."""
        if hasattr(self, "__num_simplices__"):
            return self.__num_simplices__
        if self.signal is not None:
            return self.signal.size(self.__cat_dim__("signal", self.signal))
        return None

    @num_simplices.setter
    def num_simplices(self, num_simplices):
        """Sets the number of simplices in the cochain."""
        self.__num_simplices__ = num_simplices

    @property
    def num_simplices_up(self):
        """Returns the number of simplices in the higher-dimensional cochain of co-dimension 1."""
        if hasattr(self, "__num_simplices_up__"):
            return self.__num_simplices_up__
        elif self.boundary is not None:
            return self.boundary.shape[1]
        return None

    @num_simplices_up.setter
    def num_simplices_up(self, num_simplices_up):
        """Sets the number of simplices in the higher-dimensional cochain of co-dimension 1."""
        # TODO: Add more checks here
        self.__num_simplices_up__ = num_simplices_up

    @property
    def num_simplices_down(self):
        """Returns the number of simplices in the lower-dimensional cochain."""
        if hasattr(self, "__num_simplices_down__"):
            return self.__num_simplices_down__
        return None

    @num_simplices_down.setter
    def num_simplices_down(self, num_simplices_down):
        """Sets the number of simplices in the lower-dimensional cochain."""
        self.__num_simplices_down__ = num_simplices_down

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, SparseTensor):
            # Not all apply methods are supported for `SparseTensor`, e.g.,
            # `contiguous()`. We can get around it by capturing the exception.
            try:
                return func(item)
            except AttributeError:
                return item
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        """
            Applies the function :obj:`func` to all tensor attributes
            :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
            all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        """
            Ensures a contiguous memory layout for all attributes :obj:`*keys`.
            If :obj:`*keys` is not given, all present attributes are ensured to
            have a contiguous memory layout.
        """
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        """
            Performs tensor dtype and/or device conversion to all attributes
            :obj:`*keys`.
            If :obj:`*keys` is not given, the conversion is applied to all present
            attributes.
        """
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict(
            {
                k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.__dict__.items()
            }
        )

    @property
    def mapping(self):
        return self.__mapping


class CochainBatch(Cochain):
    """A datastructure for storing a batch of cochains.

    Similarly to PyTorch Geometric, the batched cochain consists of a big cochain formed of multiple
    independent cochains on sets of disconnected cells.
    """

    def __init__(self, dim, batch=None, ptr=None, **kwargs):
        super(CochainBatch, self).__init__(dim, **kwargs)

        for key, item in kwargs.items():
            if key == "num_simplices":
                self.__num_simplices__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Cochain
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_simplices_list__ = None
        self.__num_simplices_down_list__ = None
        self.__num_simplices_up_list__ = None
        self.__num_cochains__ = None

    @classmethod
    def from_cochain_list(cls, data_list):
        """
            Constructs a batch object from a python list holding
            :class:`Cochain` objects.
            The assignment vector :obj:`batch` is created on the fly.
        """

        keys = list(set.union(*[set(data.keys) for data in data_list]))
        assert "batch" not in keys and "ptr" not in keys

        sparse_keys = [
            "boundary",
            "laplacian",
            "solenoidal",
            "irrotational",
            "coboundary",
        ]

        batch = cls.initialize_batch(data_list, keys)

        device = None
        cumsum = {
            key: [torch.tensor(0)] if key not in sparse_keys else [torch.tensor([0, 0])]
            for key in keys
        }
        cat_dims = {}
        num_simplices_list = []
        num_simplices_up_list = []
        num_simplices_down_list = []

        for i, data in enumerate(data_list):

            for key in keys:

                item = data[key]

                if item is not None:
                    # Increase values by `cumsum` value.
                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.is_sparse:
                        pass
                    elif isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    # Treat 0-dimensional tensors as 1-dimensional.
                    if isinstance(item, Tensor) and item.dim() == 0:
                        item = item.unsqueeze(0)

                    batch[key].append(item)

                    # Gather the size of the `cat` dimension.
                    cat_dim = data.__cat_dim__(key, data[key])
                    cat_dims[key] = cat_dim
                    if isinstance(item, Tensor):
                        device = item.device

                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

            if hasattr(data, "__num_simplices__"):
                num_simplices_list.append(data.__num_simplices__)
            else:
                num_simplices_list.append(None)

            if hasattr(data, "__num_simplices_up__"):
                num_simplices_up_list.append(data.__num_simplices_up__)
            else:
                num_simplices_up_list.append(None)

            if hasattr(data, "__num_simplices_down__"):
                num_simplices_down_list.append(data.__num_simplices_down__)
            else:
                num_simplices_down_list.append(None)

            num_simplices = data.num_simplices
            if num_simplices is not None:
                item = torch.full((num_simplices,), i, dtype=torch.long, device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_simplices)

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_simplices_list__ = num_simplices_list
        batch.__num_simplices_up_list__ = num_simplices_up_list
        batch.__num_simplices_down_list__ = num_simplices_down_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor) and item.is_sparse:
                # laplacian, boundary, coboundary, solenoidal, irrotational
                batch[key] = block_diagonal(*items)
            elif isinstance(item, Tensor):
                # signal, batch, complex_dim
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        return batch

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(CochainBatch, self).__getitem__(idx)
        elif isinstance(idx, int):
            raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def num_cochains(self) -> int:
        """Returns the number of cochains in the batch."""
        if self.__num_cochains__ is not None:
            return self.__num_cochains__
        return self.ptr.numel() + 1

    @classmethod
    def initialize_batch(cls, data_list, keys):
        batch = cls(data_list[0].dim)
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__num_cochains__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        return batch
