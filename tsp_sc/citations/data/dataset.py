from torch.utils.data import Dataset


class CitationsDataset(Dataset):
    def __init__(
        self, inputs, targets, components, train_indices, val_indices, test_indices
    ):
        self.num_dims = len(inputs)
        self.inputs = inputs
        self.targets = targets
        self.components = components
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def __getitem__(self, idx):

        X = {f"X{d}": self.inputs[d][idx] for d in range(self.num_dims)}
        Y = {f"Y{d}": self.targets[d][idx] for d in range(self.num_dims)}

        L = {f"L{d}": self.components["full"][d][idx] for d in range(self.num_dims)}
        I = {f"I{d}": self.components["irr"][d][idx] for d in range(self.num_dims)}
        S = {f"S{d}": self.components["sol"][d][idx] for d in range(1, self.num_dims)}

        train_indices = {
            f"train_indices_{d}": self.train_indices[d][idx]
            for d in range(self.num_dims)
        }
        val_indices = {
            f"val_indices_{d}": self.val_indices[d][idx] for d in range(self.num_dims)
        }
        test_indices = {
            f"test_indices_{d}": self.test_indices[d][idx] for d in range(self.num_dims)
        }

        batch = {
            **X,
            **Y,
            **L,
            **I,
            **S,
            **train_indices,
            **val_indices,
            **test_indices,
        }

        return batch

    def __len__(self):
        return len(self.inputs[0])
