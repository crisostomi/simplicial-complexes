import pytorch_lightning as pl


class GraphClassificationDataModule(pl.LightningDataModule):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def train_dataloader(self):
        return self.dataloader
