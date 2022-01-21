from torch.utils.data import DataLoader
from tsp_sc.graph_classification.data.collater import Collater


class ComplexDataLoader(DataLoader):
    r"""Data loader which merges cochain complexes into to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        max_dim (int): The maximum dimension of the chains to be used in the batch.
            (default: 2)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, max_dim=2, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(ComplexDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            # pin_memory=True,
            collate_fn=Collater(max_dim),
            **kwargs
        )
