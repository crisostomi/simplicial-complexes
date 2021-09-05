import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import math
import scipy.sparse.linalg
from tsp_sc.common.misc import Phases, unsqueeze_list
from tsp_sc.citations.data.dataset import CitationsDataset
from tsp_sc.common.datamodule import TopologicalDataModule


class CitationDataModule(TopologicalDataModule):
    """
        laplacians: list (considered_simplex_dim) of lists each having len (batch_size)
                     containing the laplacians (num_simplex_dim_k, num_simplex_dim_k)
                     laplacians[i][j] contains the i-dimensional Laplacian of sample j
        boundaries: list (considered_simplex_dim) of lists each having len (batch_size)
                     containing the boundaries (num_simplex_dim_k, num_simplex_dim_k+1)
                     boundaries[i][j] contains the i-dimensional boundary of sample j
        components: 'full': list (considered_simplex_dim) list[i] has len (num_complexes)
                            list[i][j] contains the i-dimensional Laplacian of sample j
                    'har': same as 'full', but containing harmonic components
                    'sol': same as 'full', but containing solenoidal components
                    'irr': same as 'full', but containing irrotational components

    """

    def __init__(self, paths, data_params):
        super(CitationDataModule, self).__init__(paths, data_params)

        self.missing_value_ratio = data_params["missing_value_ratio"]
        self.add_component_signal = data_params["add_component_signal"]
        self.component_to_add = data_params["component_to_add"]

        assert not self.add_component_signal or self.component_to_add is not None

        self.laplacians = self.load_laplacians(paths)
        self.boundaries = self.load_boundaries(paths)

        self.num_complexes = len(self.laplacians[0])
        self.num_simplices = [L[0].shape[0] for L in self.laplacians]

        self.components = self.get_orthogonal_components()

        if self.add_component_signal:
            self.basis = self.get_sol_irr_basis()

        self.normalize_components()

        # simplices is a list of dictionaries, one per dimension d
        # the dictionary's keys are the (d+1)-sets of the vertices that constitute the d-simplices
        # the dictionary's values are the indexes of the simplices in the boundary and Laplacian matrices
        simplices = np.load(paths["simplices"], allow_pickle=True)

        # known_simplices is a list of dictionaries, one per dimension d
        # the dictionary's keys are the known d-simplices
        # the dictionary's values are their indices in the Laplacian and boundaries
        self.known_simplices = np.load(paths["known_values"], allow_pickle=True,)

        # analogously missing_simplices has as keys the missing  d -simplices
        self.missing_simplices = np.load(paths["missing_values"], allow_pickle=True)

        # target is a list of dictionaries, one per dimension d.
        # The dictionary's keys are all the d -simplices.
        # The dictionary's values are the d-cochains, i.e. the number of citations of the d-simplices.
        targets = np.load(paths["citations"], allow_pickle=True)

        # input is a list of dictionaries, one per dimension d.
        # The dictionary's keys are all the d-simplices.
        # The dictionary's values are the d -cochains where the damaged portion
        # has been replaced with the median.
        inputs = np.load(paths["input_damaged"], allow_pickle=True)

        # known_indices[d] contains the list of indices (position of the simplices
        # in the Laplacian and boundary matrices) for the known simplices
        # of dimension d.
        self.known_indices = [
            list(self.known_simplices[d].values())
            for d in range(self.considered_simplex_dim + 1)
        ]

        # Analogously, missing_indices[d] contains the list
        # of indices for the missing simplices of dimension d.
        self.missing_indices = [
            list(self.missing_simplices[d].values())
            for d in range(self.considered_simplex_dim + 1)
        ]

        # Sort both input and target following the ordering of the keys in simplices
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

        self.train_indices = self.known_indices
        self.val_indices, self.test_indices = self.split_val_test(self.missing_indices)
        self.prepare_batches()

        self.targets = self.prepare_targets()
        self.inputs = self.prepare_input()

        if self.add_component_signal:
            self.add_signal_component(
                self.component_to_add, data_params["noise_energy"]
            )

        self.datasets = self.get_datasets()

        self.validate()

    def get_sol_irr_basis(self):

        irr_basis = [
            scipy.sparse.linalg.eigsh(self.components["irr"][dim][0])[1]
            for dim in range(self.considered_simplex_dim + 1)
        ]

        sol_basis = [None] + [
            scipy.sparse.linalg.eigsh(self.components["sol"][dim][0])[1]
            for dim in range(1, self.considered_simplex_dim + 1)
        ]

        basis = {"sol": sol_basis, "irr": irr_basis}
        return basis

    def add_signal_component(self, component, noise_energy):

        orthog_component = "sol" if component == "irr" else "irr"

        for dim in range(1, self.considered_simplex_dim + 1):
            n = self.num_simplices[dim]

            noise = noise_energy * np.random.rand(n) * self.inputs[dim][0].numpy()

            basis = self.basis[orthog_component][dim]

            signal_over_component = (np.identity(n) - basis @ basis.transpose()) @ noise
            signal_over_component = signal_over_component.astype("float32")

            self.inputs[dim][0] = self.inputs[dim][0] + signal_over_component
            self.targets[dim][0] = self.targets[dim][0] + signal_over_component

    def split_val_test(self, missing_indices):
        val_ratio = 0.3
        val_indices = []
        test_indices = []
        for k in range(self.considered_simplex_dim + 1):
            k_dim_missing_indices = missing_indices[k]
            random.shuffle(k_dim_missing_indices)
            val_upperbound = math.ceil(val_ratio * len(k_dim_missing_indices))
            k_dim_val_indices = k_dim_missing_indices[:val_upperbound]
            k_dim_test_indices = k_dim_missing_indices[val_upperbound:]
            val_indices.append(k_dim_val_indices)
            test_indices.append(k_dim_test_indices)

        return val_indices, test_indices

    def prepare_batches(self):
        if self.num_complexes == 1:
            self.known_indices = unsqueeze_list(self.known_indices)
            self.train_indices = unsqueeze_list(self.train_indices)
            self.val_indices = unsqueeze_list(self.val_indices)
            self.test_indices = unsqueeze_list(self.test_indices)
            self.missing_indices = unsqueeze_list(self.missing_indices)
            self.sorted_input = unsqueeze_list(self.sorted_input)
            self.sorted_target = unsqueeze_list(self.sorted_target)
            self.sorted_input_values = unsqueeze_list(self.sorted_input_values)
            self.sorted_target_values = unsqueeze_list(self.sorted_target_values)
            self.known_simplices = unsqueeze_list(self.known_simplices)

    def load_laplacians(self, paths: dict):
        """
        :param paths: dict containing the path to the laplacians
                      stored as a list (max_simplex_dim) of tensors (num_simplex_dim_k, num_simplex_dim_k)
        :return: list (considered_simplex_dim) of lists each having len (num_complexes)
                     containing the laplacians (num_simplex_dim_k, num_simplex_dim_k)
                     laplacians[i][j] contains the i-dimensional Laplacian of sample j
        """
        # laplacians[k] has shape (num_simplex_dim_k, num_simplex_dim_k)
        laplacians = np.load(paths["laplacians"], allow_pickle=True).tolist()

        # laplacians[k] has shape (num_complexes, num_simplex_dim_k, num_simplex_dim_k)
        laplacians = [[L] for L in laplacians[: self.considered_simplex_dim + 1]]
        return laplacians

    def load_boundaries(self, paths: dict):
        """
        :param paths: dict containing the path to the boundaries
                      stored as a list (max_simplex_dim) of tensors (num_simplex_dim_k, num_simplex_dim_k+1)
        :return: list (considered_simplex_dim) of lists each having len (num_complexes)
                     containing the boundaries (num_simplex_dim_k, num_simplex_dim_k+1)
                     boundaries[i][j] contains the i-dimensional boundary of sample j
        """
        # boundaries[k] has shape (num_simplex_dim_k, num_simplex_dim_k+1)
        boundaries = np.load(paths["boundaries"], allow_pickle=True).tolist()

        # boundaries[k] has shape (num_complexes, num_simplex_dim_k-1, num_simplex_dim_k)
        boundaries = [[None]] + [
            [B] for B in boundaries[: self.considered_simplex_dim + 1]
        ]
        return boundaries

    def train_dataloader(self):
        return DataLoader(self.datasets[Phases.train])

    def val_dataloader(self):
        # same dataset as training, indices are different
        return DataLoader(self.datasets[Phases.train])

    def test_dataloader(self):
        # same dataset as training, indices are different
        return DataLoader(self.datasets[Phases.train])

    def get_datasets(self):
        datasets = {
            Phases.train: CitationsDataset(
                self.inputs,
                self.targets,
                self.components,
                self.train_indices,
                self.val_indices,
                self.test_indices,
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
