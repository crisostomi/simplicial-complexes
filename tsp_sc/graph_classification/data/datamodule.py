from tsp_sc.common.datamodule import TopologicalDataModule
import numpy as np
import torch
from tsp_sc.graph_classification.data.dataset import GraphClassificationDataset
from tsp_sc.common.misc import Phases
from tsp_sc.common.simplices import (
    get_index_from_boundary,
    get_orientation_from_boundary,
)
from torch.utils.data import DataLoader
from tsp_sc.common.complex import Cochain


class GraphClassificationDataModule(TopologicalDataModule):
    def __init__(self, paths, data_params):
        super().__init__(paths, data_params)

        self.laplacians = self.load_laplacians(paths)
        self.boundaries = self.load_boundaries(paths)
        self.num_complexes = len(self.laplacians[0])
        self.complexes = self.create_complexes()

        self.components = self.get_orthogonal_components()
        self.normalize_components()

        self.inputs = self.prepare_inputs(paths)
        self.graph_indices = self.prepare_graph_indices()

        self.datasets = self.get_datasets()

    def create_complexes(self):
        cochain_list = []
        for (boundary, coboundary) in zip(
            self.boundaries[1][0:], self.boundaries[1][1:]
        ):
            cochain = Cochain.from_boundaries(
                dim=1, boundary=boundary, coboundary=coboundary
            )
            cochain_list.append(cochain)
            print(cochain.num_simplices_up)

    def load_laplacians(self, paths):
        laplacians = np.load(paths["laplacians"], allow_pickle=True)
        laplacians = [
            laplacian[: self.considered_simplex_dim + 1] for laplacian in laplacians
        ]
        laplacians = [
            [L[i] for L in laplacians] for i in range(self.considered_simplex_dim + 1)
        ]
        return laplacians

    def load_boundaries(self, paths):
        boundaries = np.load(paths["boundaries"], allow_pickle=True)
        Bs = [
            list(boundary[: self.considered_simplex_dim + 1]) for boundary in boundaries
        ]

        Bs = [None] + [[B[i] for B in Bs] for i in range(self.considered_simplex_dim)]
        return Bs

    def prepare_inputs(self, paths):
        """
        :param paths: paths to numpy array (num_complexes, 3)
        :return: list of tensors (3, num_complexes)
        """
        inputs = np.load(paths["signals"], allow_pickle=True)
        assert len(inputs) == self.num_complexes
        inputs = [
            [torch.tensor(input[i]) for input in inputs]
            for i in range(self.considered_simplex_dim + 1)
        ]

        return inputs

    def prepare_graph_indices(self):
        return None
        # graph_indices = [0]
        # current_ind = 0
        # for i in range(batch_size):
        #     # num nodes of the i-th graph
        #     num_nodes_curr_graph = all_components["full"][i][0].shape[0]
        #     current_ind += num_nodes_curr_graph
        #     graph_indices.append(current_ind)
        #
        # print(graph_indices)

    def get_datasets(self):
        datasets = {
            Phases.train: GraphClassificationDataset(self.inputs, self.components)
        }
        return datasets

    def train_dataloader(self):
        return DataLoader(self.datasets[Phases.train], batch_size=self.batch_size)

    def print_stats(self):
        num_complexes = len(self.laplacians)
        print(f"There are {num_complexes} simplicial complexes")
