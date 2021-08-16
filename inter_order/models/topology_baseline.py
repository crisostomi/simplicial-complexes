class TopologyAwareBaseline(nn.Module):
    def __init__(self, num_nodes, num_triangles, node_triangle_adj):
        """
            node_triangle_adj: tensor (num_triangles, num_nodes)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_triangles = num_triangles
        self.node_triangle_adj = node_triangle_adj.transpose(1, 0)
        # self.node_triangle_adj.requires_grad = False

        self.linear = nn.Linear(num_triangles, num_triangles)

    def forward(self, X):
        """
            X: tensor (3, num_nodes)
        """
        triangle_features = X @ self.node_triangle_adj

        # (num_triangles, num_nodes)
        triangle_features = nn.ReLU()(triangle_features)

        output = self.linear(triangle_features)

        return output.transpose(1, 0)