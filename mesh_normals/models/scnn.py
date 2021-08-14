class MySCNN(nn.Module):
    def __init__(self, filter_size, colors):
        super().__init__()

        assert colors > 0
        self.colors = colors

        num_filters = 5
        variance = 0.01
        self.num_layers = 5
        self.num_dims = 3

        self.activaction = nn.LeakyReLU()

        self.C = nn.ModuleDict({f'l{i}': nn.ModuleDict() for i in range(1, self.num_layers + 1)})
        self.aggr = nn.ModuleDict({f'l{i}': nn.ModuleDict() for i in range(1, self.num_layers + 1)})

        # layer 1
        self.C['l1']['d0'] = MySimplicialConvolution(filter_size, C_in=self.colors, C_out=self.colors * num_filters,
                                                     variance=variance)

        # layer 2
        self.C['l2']['d0'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l2']['d1'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.aggr['l2']['d1'] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU()
        )

        # layer 3
        self.C['l3']['d0'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l3']['d1'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l3']['d2'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.aggr['l3']['d1'] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU()
        )
        self.aggr['l3']['d2'] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU()
        )

        # layer 4
        self.C['l4']['d0'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l4']['d1'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l4']['d2'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.aggr['l4']['d1'] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU()
        )
        self.aggr['l4']['d2'] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU()
        )

        # layer 5
        self.C['l5']['d0'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l5']['d1'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.C['l5']['d2'] = MySimplicialConvolution(filter_size, C_in=self.colors * num_filters,
                                                     C_out=self.colors * num_filters, variance=variance)
        self.aggr['l5']['d1'] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU()
        )

        self.last_aggregator = nn.Linear(2 * self.colors * num_filters, self.colors)

    def forward(self, xs, components, Bs, Bts):
        """
        parameters:
            xs: inputs
        """

        layers = range(self.num_layers + 1)
        dims = range(self.num_dims)
        L = components['lap']

        ###### layer 1 ######

        # S0 = conv(S0)
        # (num_filters x num_dims, num_nodes)
        S0 = self.C['l1']['d0'](L[0], xs[0])
        S0 = self.activaction(S0)

        # S1 = lift(S0)
        # (num_edges, num_filters * c_in)
        S0_lifted = self.lift(Bts[0], S0)
        S1 = S0_lifted

        ###### layer 2 ######

        # S0 = conv(S0)
        # (num_filters * num_dims, num_nodes)
        S0 = self.C['l2']['d0'](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters * c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters * c_in, num_edges)
        S1_conv = self.C['l2']['d1'](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr['l2']['d1'](S1_concat)

        # S2 = lift(S1)
        S2 = Bts[1] @ S1

        ###### layer 3 ######

        # S0 = conv(S0)
        S0 = self.C['l3']['d0'](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters * c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters * c_in, num_edges)
        S1_conv = self.C['l3']['d1'](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        # (2 * num_filters * c_in, num_edges)
        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr['l3']['d1'](S1_concat)

        # (num_edges, num_filters * c_in)
        S1_lifted = Bts[1] @ S1

        # (num_filters * c_in, num_edges)
        S2_conv = self.C['l3']['d2'](L[2], S2.transpose(1, 0))
        S2_conv = self.activaction(S2_conv)

        S2_concat = torch.cat((S1_lifted, S2_conv.transpose(1, 0)), dim=1)

        S2 = self.aggr['l3']['d2'](S2_concat)

        ###### layer 4 ######

        # S0 = conv(S0)
        S0 = self.C['l4']['d0'](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters x c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters x c_in, num_edges)
        S1_conv = self.C['l4']['d1'](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr['l4']['d1'](S1_concat)

        # (num_edges, num_filters x c_in)
        S1_lifted = Bts[1] @ S1

        # (num_filters x c_in, num_edges)
        S2_conv = self.C['l4']['d2'](L[2], S2.transpose(1, 0))
        S2_conv = self.activaction(S2_conv)

        S2_concat = torch.cat((S1_lifted, S2_conv.transpose(1, 0)), dim=1)

        S2 = self.aggr['l4']['d2'](S2_concat)

        ###### layer 5 ######

        # S0 = conv(S0)
        S0 = self.C['l4']['d0'](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters x c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters x c_in, num_edges)
        S1_conv = self.C['l4']['d1'](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr['l5']['d1'](S1_concat)

        # (num_edges, num_filters x c_in)
        S1_lifted = Bts[1] @ S1

        # (num_filters x c_in, num_edges)
        S2_conv = self.C['l4']['d2'](L[2], S2.transpose(1, 0))

        S2_concat = torch.cat((S1_lifted, S2_conv.transpose(1, 0)), dim=1)

        S2 = self.last_aggregator(S2_concat)

        return S2

    def lift(self, B, S):
        return B @ S.transpose(1, 0)