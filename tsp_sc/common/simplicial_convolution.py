import torch
import torch.nn as nn


class MySimplicialConvolution(nn.Module):
    def __init__(self, filter_size, C_in, C_out, enable_bias=True, variance=1.0):
        """
        Convolution for simplices of a fixed dimension
        """
        super().__init__()

        assert C_in > 0
        assert C_out > 0
        assert filter_size > 0

        self.C_in = C_in
        self.C_out = C_out
        self.filter_size = filter_size
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(
            variance * torch.randn((self.C_out, self.C_in, self.filter_size))
        )

        self.bias = (
            nn.parameter.Parameter(torch.zeros((self.C_out, 1)))
            if self.enable_bias
            else 0.0
        )

    def forward(self, L, x):
        (channels_in, num_simplices) = x.shape

        assert channels_in == self.C_in

        X = self.my_assemble(self.filter_size, L, x)

        # X ~ channels_in, num_simplices, filter_size
        # theta ~ channels_out, channels_in, filter_size
        # y ~ channels_out, num_simplices

        y = torch.einsum("imk, oik -> om", (X, self.theta)) + self.bias

        assert y.shape == (self.C_out, num_simplices)

        return y

    @staticmethod
    def my_assemble(filter_size, L, x):
        """
        parameters:
            filter_size: Chebyshev filter size
            L: Laplacian (num_simplices, num_simplices)
            x: input (batch_size, C_in, num_simplices)
        """

        (C_in, num_simplices) = x.shape
        assert L.shape[0] == num_simplices
        assert L.shape[0] == L.shape[1]  # L is a square matrix
        assert filter_size > 0

        X = []
        # for each channel
        for c_in in range(0, C_in):
            bar_X = []
            bar_X_0 = x[c_in, :].unsqueeze(1)  # \bar{x}_0 = x
            bar_X.append(bar_X_0)

            # Chebyshev recursion
            if filter_size > 1:
                bar_X_1 = L @ bar_X[0]  # \bar{x}_1 = L x
                bar_X.append(bar_X_1)

                for k in range(2, filter_size):
                    bar_X.append(
                        2 * (L @ bar_X[k - 1]) - bar_X[k - 2]
                    )  # \bar{x}_k = 2 L \bar{x}_{k-1} - \bar{x}_{k-2}

            # (num_simplices, filter_size)
            bar_X = torch.cat(bar_X, 1)
            assert bar_X.shape == (num_simplices, filter_size)
            X.append(bar_X.unsqueeze(0))

        # (channels_in, num_simplices, filter_size)
        X = torch.cat(X, 0)
        assert X.shape == (C_in, num_simplices, filter_size)

        return X


class DeffSimplicialConvolution(nn.Module):
    def __init__(
        self, filter_size: int, C_in: int, C_out: int, enable_bias=True, variance=1.0
    ):
        """
    Convolution for simplices of a fixed dimension
    """
        super().__init__()

        assert C_in > 0
        assert C_out > 0
        assert filter_size > 0

        self.C_in = C_in
        self.C_out = C_out
        self.filter_size = filter_size
        self.enable_bias = enable_bias

        self.theta = nn.parameter.Parameter(
            variance * torch.randn((self.C_out, self.C_in, self.filter_size))
        )
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((self.C_out, 1)))
        else:
            self.bias = 0.0

    def forward(self, L, x):
        assert len(L.shape) == 2
        assert L.shape[0] == L.shape[1]

        (channels_in, num_simplices) = x.shape

        assert num_simplices == L.shape[0]
        assert channels_in == self.C_in

        X = self.assemble(self.filter_size, L, x)

        # X ~ channels_in, num_simplices, filter_size
        # theta ~ channels_out, channels_in, filter_size
        # y ~ channels_out, num_simplices

        y = torch.einsum("imk, oik -> om", (X, self.theta)) + self.bias

        assert y.shape == (self.C_out, num_simplices)

        return y

    @staticmethod
    def assemble(filter_size: int, L: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Preparates the Chebyshev polynomials
        parameters:
            filter_size: filter size
            L: Laplacian, which can be full, lower (solenoidal) or upper (irrotational) (num_simplices, num_simplices)
            x: input (batch_size, C_in, num_simplices)
        """

        (C_in, num_simplices) = x.shape

        assert L.shape[0] == num_simplices
        assert L.shape[0] == L.shape[1]  # L is a square matrix
        assert filter_size > 0

        X = []
        # for each channel
        for c_in in range(0, C_in):
            bar_X = []
            bar_X_0 = x[c_in, :].unsqueeze(1)  # \bar{x}_0 = x
            bar_X.append(bar_X_0)

            # Chebyshev recursion
            if filter_size > 1:
                bar_X_1 = L @ bar_X[0]  # \bar{x}_1 = L x
                bar_X.append(bar_X_1)

                for k in range(2, filter_size):
                    bar_X.append(
                        2 * (L @ bar_X[k - 1]) - bar_X[k - 2]
                    )  # \bar{x}_k = 2 L \bar{x}_{k-1} - \bar{x}_{k-2}

            # (num_simplices, filter_size)
            bar_X = torch.cat(bar_X, 1)
            assert bar_X.shape == (num_simplices, filter_size)
            X.append(bar_X.unsqueeze(0))

        # (channels_in, num_simplices, filter_size)
        X = torch.cat(X, 0)
        assert X.shape == (C_in, num_simplices, filter_size)

        return X
