from abc import ABC
import os.path as osp
from torch_geometric.data import Dataset


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
        """
        :param root: root folder where data are stored
        :param transform: dynamically transforms the data object before accessing
        :param pre_transform: applies the transformation before saving the data objects to disk
        :param pre_filter: manually filter out data objects before saving
        :param max_dim: maximum dimension of the considered chains
        :param num_classes: number of classes/labels in the dataset
        :param init_method: aggregation method for the initialization of simplex features
        """
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
        :return: size of the features in dimension dim
        """
        if dim > self.max_dim:
            raise ValueError(
                f"dim {dim} larger than max allowed dimension {self.max_dim}."
            )
        if self._num_features[dim] is None:
            self._look_up_num_features()
        return self._num_features[dim]

    def num_features(self) -> dict:
        """
        :return: dictionary containing for each dimension its feature size
        """
        for dim in range(self.max_dim):
            if self._num_features[dim] is None:
                self._look_up_num_features()
        return self._num_features

    def _look_up_num_features(self):
        """
        Look up feature size for each dimension. All complexes must have
        the same feature size for the same dimension.
        """
        # get a complex which has all the considered dimensions
        largest_complex = self.get_complex_with_max_dim()

        for dim in range(self.max_dim + 1):
            if self._num_features[dim] is None:
                self._num_features[dim] = largest_complex.cochains[dim].num_features

    def get_complex_with_max_dim(self):
        """
        :return: any complex which has all the considered dimensions
        """
        for complex in self:
            if complex.dimension == self.max_dim:
                return complex
        raise AssertionError(f"no complex having dimension {self.max_dim}")

    def get_idx_split(self):
        """
        :return: dict containing for each stage (train, valid, test) its indices
        """
        idx_split = {
            "train": self.train_ids,
            "valid": self.val_ids,
            "test": self.test_ids,
        }
        return idx_split
