class LinearBaseline(nn.Module):
    def __init__(self, num_nodes, num_triangles):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_triangles = num_triangles

        self.node_to_triangles = nn.Linear(self.num_nodes, self.num_triangles, dtype=float)
        self.linear = nn.Linear(self.num_triangles, self.num_triangles, dtype=float)

    def forward(self, X):
        """
        """
        triangle_features = self.node_to_triangles(X)

        triangle_features = nn.ReLU()(triangle_features)

        output = self.linear(triangle_features)

        return output.transpose(1, 0)