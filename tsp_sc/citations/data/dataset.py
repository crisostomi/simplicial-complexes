from torch.utils.data import Dataset


class CitationsDataset(Dataset):
    def __init__(
        self, inputs, targets, components, train_indices, val_indices, test_indices
    ):
        self.inputs = inputs
        self.targets = targets
        self.components = components
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

    def __getitem__(self, idx):
        return {
            "X0": self.inputs[0][idx],
            "X1": self.inputs[1][idx],
            "X2": self.inputs[2][idx],
            "Y0": self.targets[0][idx],
            "Y1": self.targets[1][idx],
            "Y2": self.targets[2][idx],
            "train_indices_0": self.train_indices[0][idx],
            "train_indices_1": self.train_indices[1][idx],
            "train_indices_2": self.train_indices[2][idx],
            "val_indices_0": self.val_indices[0][idx],
            "val_indices_1": self.val_indices[1][idx],
            "val_indices_2": self.val_indices[2][idx],
            "test_indices_0": self.test_indices[0][idx],
            "test_indices_1": self.test_indices[1][idx],
            "test_indices_2": self.test_indices[2][idx],
            "L0": self.components["full"][0][idx],
            "L1": self.components["full"][1][idx],
            "L2": self.components["full"][2][idx],
            "I0": self.components["irr"][0][idx],
            "I1": self.components["irr"][1][idx],
            "I2": self.components["irr"][2][idx],
            "S1": self.components["sol"][1][idx],
            "S2": self.components["sol"][2][idx],
        }

    def __len__(self):
        return len(self.inputs[0])
