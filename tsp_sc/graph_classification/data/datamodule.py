import pytorch_lightning as pl
from tsp_sc.graph_classification.data.dataloader import ComplexDataLoader
from tsp_sc.common.misc import Phases
from tsp_sc.graph_classification.utils import load_dataset


class GraphClassificationDataModule(pl.LightningDataModule):
    def __init__(self, paths, data_params):
        super().__init__()

        self.dataset = load_dataset(
            name=data_params["dataset"],
            root=paths["data"],
            max_dim=data_params["considered_simplex_dim"],
            init_method=data_params["init_method"],
            fold=data_params["fold"],
        )

        self.batch_size = data_params["batch_size"]

    def train_dataloader(self):
        return ComplexDataLoader(
            self.dataset[self.dataset.split_indices[Phases.train]],
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return ComplexDataLoader(
            self.dataset[self.dataset.split_indices[Phases.val]],
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        if Phases.test in self.dataset.split_indices:
            return ComplexDataLoader(
                self.dataset[self.dataset.split_indices[Phases.test]],
                batch_size=self.batch_size,
                shuffle=False,
            )
        return None
