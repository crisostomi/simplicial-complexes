from scipy.sparse import coo_matrix
from scipy.linalg import null_space
from tsp_sc.common.simplices import normalize_laplacian
from tsp_sc.common.misc import coo2tensor
from scipy.sparse import linalg
import torch
from typing import List
from torch import Tensor
from torch_sparse import SparseTensor
from tsp_sc.common.utils import block_diagonal
import logging
import copy


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

        if self.dim > 0 and self.dim < self.complex_dim:
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
        return torch.sparse_coo_tensor(indices=indices, values=data)

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
            row_inc, col_inc = self.num_simplices, self.num_simplices
            inc = [row_inc, col_inc]
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
        return 0

    @num_simplices_up.setter
    def num_simplices_up(self, num_simplices_up):
        """Sets the number of simplices in the higher-dimensional cochain of co-dimension 1."""
        # TODO: Add more checks here
        self.__num_simplices_up__ = num_simplices_up

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
    def from_cochain_list(cls, data_list, follow_batch=[]):
        """
            Constructs a batch object from a python list holding
            :class:`Cochain` objects.
            The assignment vector :obj:`batch` is created on the fly.
            Additionally, creates assignment batch vectors for each key in
            :obj:`follow_batch`.
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

        batch = cls(data_list[0].dim)
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__num_cochains__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        device = None
        slices = {key: [0] for key in keys}
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
                    size = 1
                    cat_dim = data.__cat_dim__(key, data[key])
                    cat_dims[key] = cat_dim
                    if isinstance(item, Tensor):
                        if item.is_sparse:
                            size = torch.tensor(item.size())[torch.tensor(cat_dim)]
                            device = item.device
                        else:
                            size = item.size(cat_dim)
                            device = item.device

                    # TODO: do we really need slices, and, are we managing them correctly?
                    slices[key].append(size + slices[key][-1])

                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f"{key}_{j}_batch"
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full(
                                        (size,), i, dtype=torch.long, device=device
                                    )
                                )
                        else:
                            tmp = f"{key}_batch"
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size,), i, dtype=torch.long, device=device)
                            )

                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                    print(inc)
                cumsum[key].append(inc + cumsum[key][-1])
                print(cumsum)

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

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
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
                batch[key] = block_diagonal(*items)
            elif isinstance(item, Tensor):
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


class SimplicialComplex:
    """Class representing a cochain complex

    Args:
        cochains: A list of cochains forming the cochain complex
        y: A tensor of shape (1,) containing a label for the complex for complex-level tasks.
        dimension: The dimension of the complex.
    """

    def __init__(
        self, *cochains: Cochain, y: torch.Tensor = None, dimension: int = None
    ):
        if len(cochains) == 0:
            raise ValueError("At least one cochain is required.")
        if dimension is None:
            dimension = len(cochains) - 1
        if len(cochains) < dimension:
            raise ValueError(
                f"Not enough cochains passed, "
                f"expected {dimension + 1}, received {len(cochains)}"
            )

        self.dimension = dimension
        self.cochains = {i: cochains[i] for i in range(dimension + 1)}
        self.nodes = cochains[0]
        self.edges = cochains[1] if dimension >= 1 else None
        self.triangles = cochains[2] if dimension >= 2 else None

        self.y = y

    def to(self, device, **kwargs):
        """Performs tensor dtype and/or device conversion to cochains and label y, if set."""
        # TODO: handle device conversion for specific attributes via `*keys` parameter
        for dim in range(self.dimension + 1):
            self.cochains[dim] = self.cochains[dim].to(device, **kwargs)
        if self.y is not None:
            self.y = self.y.to(device, **kwargs)
        return self

    def get_labels(self, dim=None):
        """Returns target labels.

        If `dim`==k (integer in [0, self.dimension]) then the labels over k-cells are returned.
        In the case `dim` is None the complex-wise label is returned.
        """
        if dim is None:
            y = self.y
        else:
            if dim in self.cochains:
                y = self.cochains[dim].y
            else:
                raise NotImplementedError(
                    "Dim {} is not present in the complex or not yet supported.".format(
                        dim
                    )
                )
        return y

    def set_signals(self, signals: List[Tensor]):
        """Sets the features of the cochains to the values in the list"""
        assert (self.dimension + 1) >= len(signals)
        for i, signal in enumerate(signals):
            self.cochains[i].signal = signal

    @property
    def keys(self):
        """Returns all names of complex attributes."""
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


class ComplexBatch(SimplicialComplex):
    """Class representing a batch of cochain complexes.

    This is stored as a single cochain complex formed of batched cochains.

    Args:
        cochains: A list of cochain batches that will be put together in a complex batch
        dimension: The dimension of the resulting complex.
        y: A tensor of labels for the complexes in the batch.
        num_complexes: The number of complexes in the batch.
    """

    def __init__(
        self,
        *cochains: CochainBatch,
        dimension: int,
        y: torch.Tensor = None,
        num_complexes: int = None,
    ):
        super(ComplexBatch, self).__init__(*cochains, y=y)
        self.num_complexes = num_complexes
        self.dimension = dimension

    @classmethod
    def from_complex_list(
        cls, data_list: List[SimplicialComplex], follow_batch=[], max_dim: int = 2
    ):
        """Constructs a ComplexBatch from a list of complexes.

        Args:
            data_list: a list of complexes from which the batch is built.
            follow_batch: creates assignment batch vectors for each key in
                :obj:`follow_batch`.
            max_dim: the maximum cochain dimension considered when constructing the batch.
        Returns:
            A ComplexBatch object.
        """

        dimension = max([data.dimension for data in data_list])
        dimension = min(dimension, max_dim)
        cochains = [list() for _ in range(dimension + 1)]
        label_list = list()
        per_complex_labels = True

        for comp in data_list:
            for dim in range(dimension + 1):
                if dim not in comp.cochains:
                    # If a dim-cochain is not present for the current complex, we instantiate one.
                    cochains[dim].append(Cochain(dim=dim))
                    if dim - 1 in comp.cochains:
                        # If the cochain below exists in the complex, we need to add the number of
                        # boundaries to the newly initialised complex, otherwise batching will not work.
                        cochains[dim][-1].num_simplices_down = comp.cochains[
                            dim - 1
                        ].num_simplices
                else:
                    cochains[dim].append(comp.cochains[dim])
            per_complex_labels &= comp.y is not None
            if per_complex_labels:
                label_list.append(comp.y)

        batched_cochains = [
            CochainBatch.from_cochain_list(cochain_list, follow_batch=follow_batch)
            for cochain_list in cochains
        ]
        y = None if not per_complex_labels else torch.cat(label_list, 0)
        batch = cls(
            *batched_cochains, y=y, num_complexes=len(data_list), dimension=dimension
        )

        return batch
