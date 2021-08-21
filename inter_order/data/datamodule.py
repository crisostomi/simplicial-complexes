import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from inter_order.utils.misc import coo2tensor, Phases
from inter_order.utils.simplices import normalize_laplacian
from inter_order.data.dataset import NormalsDataset


class NormalsDataModule(pl.LightningDataModule):
    def __init__(self, data):
        super(NormalsDataModule, self).__init__()

        self.laplacians, self.boundaries = data["laplacians"], data["boundaries"]
        self.num_nodes, self.num_edges, self.num_triangles = [
            L.shape[0] for L in self.laplacians
        ]
        self.original_positions, self.triangles = (
            data["original_positions"],
            data["triangles"],
        )
        self.noisy_positions = data["noisy_positions"]

        max_simplex_dim = 2
        self.normalized_laplacians = [
            [
                coo2tensor(normalize_laplacian(self.laplacians[i], half_interval=True))
                for i in range(max_simplex_dim + 1)
            ]
        ]

        self.boundaries = [
            [None] + [coo2tensor(self.boundaries[i]) for i in range(max_simplex_dim)]
        ]

        self.node_positions = self.prepare_node_positions(data["positions"])
        self.triangle_normals = self.prepare_triangle_normals(data["normals"])
        self.datasets = self.get_datasets()

    def prepare_node_positions(self, node_positions):
        node_signal = [
            torch.tensor(position).float() for position in list(node_positions.values())
        ]
        node_signal = torch.stack(node_signal).transpose(1, 0)

        return [node_signal]

    def prepare_triangle_normals(self, triangle_normals):
        targets = [
            torch.tensor(signal).float() for signal in list(triangle_normals.values())
        ]
        targets = torch.stack(targets)
        return [targets]

    def get_datasets(self):
        train_dataset = NormalsDataset(
            self.node_positions,
            self.triangle_normals,
            self.normalized_laplacians,
            self.boundaries,
        )
        # TODO: add a val/test dataset
        datasets = {
            Phases.train: train_dataset,
            Phases.val: None,
            Phases.test: train_dataset,
        }
        return datasets

    def train_dataloader(self):
        return DataLoader(self.datasets[Phases.train])

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.datasets[Phases.test])

    def print_stats(self):
        print(
            f"There are {self.num_nodes} nodes, {self.num_edges} edges and {self.num_triangles} triangles"
        )
