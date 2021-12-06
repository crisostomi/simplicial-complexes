from tsp_sc.common.simp_complex import (
    Cochain,
    SimplicialComplex,
    ComplexBatch,
    CochainBatch,
)


class Collater(object):
    """Object that converts python lists of objects into the appropiate storage format.

    Args:
        follow_batch: Creates assignment batch vectors for each key in the list.
        max_dim: The maximum dimension of the cochains considered from the supplied list.
    """

    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        """Converts a data list in the right storage format."""
        elem = batch[0]

        if isinstance(elem, SimplicialComplex):
            return ComplexBatch.from_complex_list(batch, max_dim=self.max_dim)

        raise TypeError("DataLoader found invalid type: {}".format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)
