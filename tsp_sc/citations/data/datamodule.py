import torch
from torch.utils.data import DataLoader
import numpy as np

from tsp_sc.common.misc import Phases
from tsp_sc.citations.data.dataset import CitationsDataset
from tsp_sc.common.datamodule import TopologicalDataModule


class CitationDataModule(TopologicalDataModule):
    def __init__(self, paths, data_params):
        super(CitationDataModule, self).__init__(paths, data_params)

        self.missing_value_ratio = data_params["missing_value_ratio"]

        self.laplacians = self.load_laplacians(paths)
        self.boundaries = self.load_boundaries(paths)

        self.num_simplices = [L[0].shape[0] for L in self.laplacians]

        self.components = self.get_orthogonal_components()

        self.normalize_components()

        simplices = np.load(paths["simplices"], allow_pickle=True)

        self.known_simplices = np.load(paths["known_values"], allow_pickle=True,)
        self.missing_simplices = np.load(paths["missing_values"], allow_pickle=True)
        targets = np.load(paths["citations"], allow_pickle=True)
        inputs = np.load(paths["input_damaged"], allow_pickle=True)

        self.known_indices = [
            list(self.known_simplices[d].values())
            for d in range(self.considered_simplex_dim + 1)
        ]
        self.missing_indices = [
            list(self.missing_simplices[d].values())
            for d in range(self.considered_simplex_dim + 1)
        ]

        self.sorted_input = [
            {key: int(inputs[k][key]) for key, _ in simplices[k].items()}
            for k in range(0, self.considered_simplex_dim + 1)
        ]
        self.sorted_target = [
            {key: int(targets[k][key]) for key, _ in simplices[k].items()}
            for k in range(0, self.considered_simplex_dim + 1)
        ]
        self.sorted_input_values = [
            list(self.sorted_input[k].values())
            for k in range(0, self.considered_simplex_dim + 1)
        ]
        self.sorted_target_values = [
            list(self.sorted_target[k].values())
            for k in range(0, self.considered_simplex_dim + 1)
        ]

        self.prepare_batches()

        self.targets = self.prepare_targets()
        self.inputs = self.prepare_input()

        self.datasets = self.get_datasets()

        self.validate()

    def prepare_batches(self):
        if self.batch_size == 1:
            self.known_indices = self.unsqueeze_list(self.known_indices)
            self.missing_indices = self.unsqueeze_list(self.missing_indices)
            self.sorted_input = self.unsqueeze_list(self.sorted_input)
            self.sorted_target = self.unsqueeze_list(self.sorted_target)
            self.sorted_input_values = self.unsqueeze_list(self.sorted_target_values)
            self.sorted_target_values = self.unsqueeze_list(self.sorted_target_values)
            self.known_simplices = self.unsqueeze_list(self.known_simplices)

    def load_laplacians(self, paths):
        # laplacians[k] has shape (num_simplex_dim_k, num_simplex_dim_k)
        laplacians = np.load(paths["laplacians"], allow_pickle=True).tolist()

        # laplacians[k] has shape (1, num_simplex_dim_k, num_simplex_dim_k)
        laplacians = [[L] for L in laplacians[: self.considered_simplex_dim + 1]]
        return laplacians

    def load_boundaries(self, paths):
        # boundaries[k] has shape (num_simplex_dim_k, num_simplex_dim_k+1)
        boundaries = np.load(paths["boundaries"], allow_pickle=True).tolist()

        # TODO: handle batching
        # boundaries[k] has shape (1, num_simplex_dim_k-1, num_simplex_dim_k)
        boundaries = [None] + [
            [B] for B in boundaries[: self.considered_simplex_dim + 1]
        ]
        return boundaries

    def train_dataloader(self):
        return DataLoader(self.datasets[Phases.train])

    def test_dataloader(self):
        # definetely weird to return the training dataset,
        # but the loss is computed on different indices
        return DataLoader(self.datasets[Phases.train])

    def get_datasets(self):
        datasets = {
            Phases.train: CitationsDataset(
                self.inputs, self.targets, self.components, self.known_indices
            )
        }
        return datasets

    def prepare_targets(self):
        """
        :return: targets[k] is a list (batch_size) of tensors (num_simplex_dim_k)
        """
        targets = []

        # TODO: handle batching
        for k in range(0, self.considered_simplex_dim + 1):
            batch_targets = []
            for b in range(0, self.batch_size):
                # shape (num_simplices_dim_k)
                target = torch.tensor(
                    self.sorted_target_values[k][b],
                    dtype=torch.float,
                    requires_grad=False,
                )
                batch_targets.append(target)
            targets.append(batch_targets)

        return targets

    def prepare_input(self):
        """
        :return: inputs[k] is a list (batch_size) of tensors (num_simplex_dim_k)
        """
        inputs = []

        # TODO: handle batching
        for k in range(0, self.considered_simplex_dim + 1):
            batch_inputs = []
            for b in range(self.batch_size):
                # shape (num_simplices_dim_k)
                input = torch.tensor(
                    self.sorted_input_values[k][b],
                    dtype=torch.float,
                    requires_grad=False,
                )
                batch_inputs.append(input)

            inputs.append(batch_inputs)

        return inputs

    def validate(self):
        self.validate_decomposition()

        self.check_input_target_consistency()

        self.check_missing_value_ratio()

    def validate_decomposition(self):
        # the first irrotational component must be equal to the node Laplacian
        tol = 1e-6
        num_nodes = len(self.components["irr"][0])

        comparison = (
            np.abs(
                self.components["irr"][0][0].cpu().to_dense()
                - self.components["full"][0][0].cpu().to_dense()
            )
            <= tol
        )
        assert comparison.all()

        # the sum of the first solenoidal component and the first irrotational component
        # must be equal to the edge Laplacian. This does not happen as both solenoidal and irrotational
        # components are normalized separately
        res = self.components["irr"][1] + self.components["sol"][1]

        # # the sum of the last solenoidal component and the last irrotational component
        # must be equal to the triangle Laplacian. Again, this does not happen as both
        # solenoidal and irrotational components are normalized separately
        res = (
            self.components["irr"][self.considered_simplex_dim][0]
            + self.components["sol"][self.considered_simplex_dim][0]
        )

    def check_input_target_consistency(self):
        for k in range(self.considered_simplex_dim + 1):
            for b in range(self.batch_size):
                known_simplices_keys = list(self.known_simplices[k][b].keys())
                known_simplices_indices = list(self.known_simplices[k][b].values())

                for i in range(len(self.known_simplices[k][b])):
                    key = known_simplices_keys[i]
                    assert self.sorted_input[k][b][key] == self.sorted_target[k][b][key]

                    index = known_simplices_indices[i]
                    assert (
                        self.sorted_input_values[k][b][index]
                        == self.sorted_target_values[k][b][index]
                    )

        for k in range(0, self.considered_simplex_dim + 1):
            for b in range(self.batch_size):
                assert len(self.inputs[k][b]) == len(self.targets[k][b])

    def check_missing_value_ratio(self):
        avg_known_indices_ratio = sum(
            [
                float(len(self.known_indices[d][0]))
                / float(self.targets[d][0].shape[0])
                for d in range(0, self.considered_simplex_dim + 1)
            ]
        ) / (self.considered_simplex_dim + 1)
        tol = 5e-2
        expected_known_indices_ratio = 1 - (self.missing_value_ratio / 100)
        assert (
            expected_known_indices_ratio - tol
            <= avg_known_indices_ratio
            <= expected_known_indices_ratio + tol
        )

    @staticmethod
    def unsqueeze_list(l):
        return [[elem] for elem in l]
