def project_signal_component(self, component):
    orthog_component = "sol" if component == "irr" else "irr"
    print(f"Projecting over {component}, so orthogonally to {orthog_component}")

    for dim in range(1, self.considered_simplex_dim + 1):
        n = self.num_simplices[dim]

        orthog_basis = self.basis[orthog_component][dim]

        projector = np.identity(n) - orthog_basis @ orthog_basis.transpose()
        signal_over_component = projector @ self.inputs[dim][0].numpy()
        target_over_component = projector @ self.targets[dim][0].numpy()

        # print(projected_over_other_comp)
        # tol = 1e-4
        # comparison = np.abs(signal_over_component - projected_twice) <= tol
        # assert comparison.all()

        self.inputs[dim][0] = torch.tensor(signal_over_component.astype("float32"))
        self.targets[dim][0] = torch.tensor(target_over_component.astype("float32"))


from tsp_sc.common.datamodule import TopologicalDataModule
import numpy as np
import torch
from tsp_sc.graph_classification.data.dataset import InMemoryComplexDataset
from tsp_sc.common.misc import Phases
from tsp_sc.common.simplices import (
    get_index_from_boundary,
    get_orientation_from_boundary,
)
from torch.utils.data import DataLoader
from tsp_sc.common.simp_complex import SimplicialComplex, Cochain


class GraphClassificationDataModule(TopologicalDataModule):
    def __init__(self, paths, data_params):
        super().__init__(paths, data_params)

        self.paths = paths

        self.complexes = self.get_complexes()
        self.num_complexes = len(self.complexes)

        # self.graph_indices = self.prepare_graph_indices()

        self.datasets = self.get_datasets()

    def get_complexes(self):
        complex_list = []
        all_laplacians = np.load(self.paths["laplacians"], allow_pickle=True)
        all_boundaries = np.load(self.paths["boundaries"], allow_pickle=True)
        all_signals = np.load(self.paths["signals"], allow_pickle=True)

        for laplacians, boundaries, signals in zip(
            all_laplacians, all_boundaries, all_signals
        ):

            coboundaries = [
                b.transpose() for b in boundaries[1 : self.considered_simplex_dim + 1]
            ]
            boundaries = [None] + [boundaries[: self.considered_simplex_dim - 1]]

            cochains = []

            for dim, (laplacian, boundary, coboundary, signal) in enumerate(
                zip(laplacians, boundaries, coboundaries, signals)
            ):
                cochain = Cochain(
                    laplacian=laplacian,
                    boundary=boundary,
                    coboundary=coboundary,
                    signal=signal,
                    dim=dim,
                )
                cochains.append(cochain)

            complex = SimplicialComplex(
                cochains=cochains, complex_dim=self.considered_simplex_dim
            )

            complex_list.append(complex)

        return complex_list

    # def load_laplacians(self, paths):
    #     laplacians = np.load(paths["laplacians"], allow_pickle=True)
    #     laplacians = [
    #         laplacian[: self.considered_simplex_dim + 1] for laplacian in laplacians
    #     ]
    #     laplacians = [
    #         [L[i] for L in laplacians] for i in range(self.considered_simplex_dim + 1)
    #     ]
    #     return laplacians
    #
    # def load_boundaries(self, paths):
    #     boundaries = np.load(paths["boundaries"], allow_pickle=True)
    #     Bs = [
    #         list(boundary[: self.considered_simplex_dim + 1]) for boundary in boundaries
    #     ]
    #
    #     Bs = [None] + [[B[i] for B in Bs] for i in range(self.considered_simplex_dim)]
    #     return Bs

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
        datasets = {Phases.train: GraphClassificationDataset(self.complexes)}
        return datasets

    def train_dataloader(self):
        return DataLoader(self.datasets[Phases.train], batch_size=self.batch_size)

    def print_stats(self):
        num_complexes = len(self.laplacians)
        print(f"There are {num_complexes} simplicial complexes")
