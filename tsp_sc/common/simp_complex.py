from tsp_sc.common.cochain import Cochain, CochainBatch
import torch
from typing import List
from torch import Tensor


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
        self._validate_input(cochains, dimension)

        self.dimension = dimension

        self.cochains = {i: cochains[i] for i in range(dimension + 1)}

        self.nodes = cochains[0]
        self.edges = cochains[1] if dimension >= 1 else None
        self.triangles = cochains[2] if dimension >= 2 else None

        self.y = y

        self._consolidate()

    @staticmethod
    def _validate_input(cochains, dimension):
        if len(cochains) == 0:
            raise ValueError("At least one cochain is required.")
        if dimension is None:
            dimension = len(cochains) - 1
        if len(cochains) < dimension:
            raise ValueError(
                f"Not enough cochains passed, "
                f"expected {dimension + 1}, received {len(cochains)}"
            )

    def _consolidate(self):
        for dim in range(self.dimension + 1):
            cochain = self.cochains[dim]
            assert cochain.dim == dim
            if dim < self.dimension:
                upper_cochain = self.cochains[dim + 1]
                num_simplices_up = upper_cochain.num_simplices
                assert num_simplices_up is not None
                if "num_simplices_up" in cochain:
                    assert cochain.num_simplices_up == num_simplices_up
                else:
                    cochain.num_simplices_up = num_simplices_up
            if dim > 0:
                lower_cochain = self.cochains[dim - 1]
                num_simplices_down = lower_cochain.num_simplices
                assert num_simplices_down is not None
                if "num_simplices_down" in cochain:
                    assert cochain.num_simplices_down == num_simplices_down
                else:
                    cochain.num_simplices_down = num_simplices_down

    def to(self, device, **kwargs):
        """Performs tensor dtype and/or device conversion to cochains and label y, if set."""
        for dim in range(self.dimension + 1):
            self.cochains[dim] = self.cochains[dim].to(device, **kwargs)
        if self.y is not None:
            self.y = self.y.to(device, **kwargs)
        return self

    def get_labels(self, dim=None):
        """
        Returns target labels.

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
        super(ComplexBatch, self).__init__(*cochains, y=y, dimension=dimension)
        self.num_complexes = num_complexes
        self.dimension = dimension

    def pin_memory(self):
        for cochain in self.cochains.values():
            cochain.pin_memory()

    @classmethod
    def from_complex_list(cls, data_list: List[SimplicialComplex], max_dim: int = 2):
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
            CochainBatch.from_cochain_list(cochain_list) for cochain_list in cochains
        ]
        y = None if not per_complex_labels else torch.cat(label_list, 0)
        batch = cls(
            *batched_cochains, y=y, num_complexes=len(data_list), dimension=dimension
        )

        return batch
