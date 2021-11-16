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
