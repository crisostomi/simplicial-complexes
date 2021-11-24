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
        # if isinstance(elem, Cochain):
        #     return CochainBatch.from_cochain_list(batch, self.follow_batch)
        if isinstance(elem, SimplicialComplex):
            return ComplexBatch.from_complex_list(
                batch, self.follow_batch, max_dim=self.max_dim
            )
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError("DataLoader found invalid type: {}".format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)
