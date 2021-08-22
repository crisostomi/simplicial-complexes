from torch.utils.data import Dataset


class NormalsDataset(Dataset):
    def __init__(self, node_positions, triangle_normals, laplacians, boundaries):
        self.node_positions = node_positions
        self.triangle_normals = triangle_normals
        self.laplacians = laplacians
        self.boundaries = boundaries

    def __getitem__(self, idx):
        return {
            "node_positions": self.node_positions[idx],
            "triangle_normals": self.triangle_normals[idx],
            "node_laplacian": self.laplacians[idx][0],
            "edge_laplacian": self.laplacians[idx][1],
            "triangle_laplacian": self.laplacians[idx][2],
            "B1": self.boundaries[idx][1],
            "B2": self.boundaries[idx][2],
        }

    def __len__(self):
        return len(self.laplacians)
