from torch.utils.data import Dataset


class GraphClassificationDataset(Dataset):
    def __init__(self, inputs, components):
        self.inputs = inputs
        self.components = components

    def __getitem__(self, idx):
        return {
            "X0": self.inputs[0][idx],
            "X1": self.inputs[1][idx],
            "X2": self.inputs[2][idx],
            "L0": self.components["full"][0][idx],
            "L1": self.components["full"][1][idx],
            "L2": self.components["full"][2][idx],
            "I0": self.components["irr"][0][idx],
            "I1": self.components["irr"][1][idx],
            "S1": self.components["sol"][1][idx],
            "S2": self.components["sol"][2][idx],
        }

    def __len__(self):
        return len(self.inputs[0])
