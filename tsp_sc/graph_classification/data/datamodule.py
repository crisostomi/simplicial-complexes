from tsp_sc.common.datamodule import TopologicalDataModule
import numpy as np
import torch


class GraphClassificationDataModule(TopologicalDataModule):
    def __init__(self, paths, data_params):
        super().__init__(paths, data_params)

        self.laplacians = self.load_laplacians(paths)
        self.boundaries = self.load_boundaries(paths)

        print(len(self.boundaries))
        self.components = self.get_orthogonal_components()

        self.normalize_components()

        self.inputs = self.prepare_inputs(paths)

        self.graph_indices = self.prepare_graph_indices()

    def load_laplacians(self, paths):
        laplacians = np.load(paths["laplacians"], allow_pickle=True)
        laplacians = [
            laplacian[: self.considered_simplex_dim + 1] for laplacian in laplacians
        ]
        laplacians = (
            [[L[0] for L in laplacians]]
            + [[L[1] for L in laplacians]]
            + [[L[2] for L in laplacians]]
        )
        return laplacians

    def load_boundaries(self, paths):
        boundaries = np.load(paths["boundaries"], allow_pickle=True)
        Bs = [
            list(boundary[: self.considered_simplex_dim + 1]) for boundary in boundaries
        ]

        Bs = [None] + [[B[0] for B in Bs]] + [[B[1] for B in Bs]]
        return Bs

    def prepare_inputs(self, paths):
        inputs = np.load(paths["inputs"], allow_pickle=True)
        inputs = [torch.tensor(input[0]) for input in inputs]

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

    def print_stats(self):
        num_complexes = len(self.laplacians)
        print(f"There are {num_complexes} simplicial complexes")
