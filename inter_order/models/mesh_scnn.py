import torch
import torch.nn as nn
from inter_order.models.simplicial_convolution import MySimplicialConvolution
import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn.functional import normalize
from inter_order.utils.misc import compute_angle_diff, compute_per_coord_diff


class MeshSCNN(pl.LightningModule):
    def __init__(self, params, plotter):
        super().__init__()
        filter_size, colors = params["filter_size"], params["colors"]
        self.plotter = plotter

        assert colors > 0
        self.colors = colors

        num_filters = 5
        variance = 0.01
        self.num_layers = 5
        self.num_dims = 3

        self.activaction = nn.LeakyReLU()

        self.C = nn.ModuleDict(
            {f"l{i}": nn.ModuleDict() for i in range(1, self.num_layers + 1)}
        )
        self.aggr = nn.ModuleDict(
            {f"l{i}": nn.ModuleDict() for i in range(1, self.num_layers + 1)}
        )

        # layer 1
        self.C["l1"]["d0"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors,
            C_out=self.colors * num_filters,
            variance=variance,
        )

        # layer 2
        self.C["l2"]["d0"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l2"]["d1"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.aggr["l2"]["d1"] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU(),
        )

        # layer 3
        self.C["l3"]["d0"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l3"]["d1"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l3"]["d2"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.aggr["l3"]["d1"] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU(),
        )
        self.aggr["l3"]["d2"] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU(),
        )

        # layer 4
        self.C["l4"]["d0"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l4"]["d1"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l4"]["d2"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.aggr["l4"]["d1"] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU(),
        )
        self.aggr["l4"]["d2"] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU(),
        )

        # layer 5
        self.C["l5"]["d0"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l5"]["d1"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.C["l5"]["d2"] = MySimplicialConvolution(
            filter_size,
            C_in=self.colors * num_filters,
            C_out=self.colors * num_filters,
            variance=variance,
        )
        self.aggr["l5"]["d1"] = nn.Sequential(
            nn.Linear(2 * self.colors * num_filters, self.colors * num_filters),
            nn.ReLU(),
        )

        self.last_aggregator = nn.Linear(2 * self.colors * num_filters, self.colors)

    def forward(self, X, boundaries, laplacians):
        """
        parameters:
            xs: inputs
        """

        layers = range(self.num_layers + 1)
        dims = range(self.num_dims)

        L = laplacians
        Bs = boundaries
        Bts = [B.transpose(1, 0) for B in Bs]

        ###### layer 1 ######

        # S0 = conv(S0)
        # (num_filters x num_dims, num_nodes)
        S0 = self.C["l1"]["d0"](L[0], X)
        S0 = self.activaction(S0)

        # S1 = lift(S0)
        # (num_edges, num_filters * c_in)
        S0_lifted = self.lift(Bts[0], S0)
        S1 = S0_lifted

        ###### layer 2 ######

        # S0 = conv(S0)
        # (num_filters * num_dims, num_nodes)
        S0 = self.C["l2"]["d0"](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters * c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters * c_in, num_edges)
        S1_conv = self.C["l2"]["d1"](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr["l2"]["d1"](S1_concat)

        # S2 = lift(S1)
        S2 = Bts[1] @ S1

        ###### layer 3 ######

        # S0 = conv(S0)
        S0 = self.C["l3"]["d0"](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters * c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters * c_in, num_edges)
        S1_conv = self.C["l3"]["d1"](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        # (2 * num_filters * c_in, num_edges)
        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr["l3"]["d1"](S1_concat)

        # (num_edges, num_filters * c_in)
        S1_lifted = Bts[1] @ S1

        # (num_filters * c_in, num_edges)
        S2_conv = self.C["l3"]["d2"](L[2], S2.transpose(1, 0))
        S2_conv = self.activaction(S2_conv)

        S2_concat = torch.cat((S1_lifted, S2_conv.transpose(1, 0)), dim=1)

        S2 = self.aggr["l3"]["d2"](S2_concat)

        ###### layer 4 ######

        # S0 = conv(S0)
        S0 = self.C["l4"]["d0"](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters x c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters x c_in, num_edges)
        S1_conv = self.C["l4"]["d1"](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr["l4"]["d1"](S1_concat)

        # (num_edges, num_filters x c_in)
        S1_lifted = Bts[1] @ S1

        # (num_filters x c_in, num_edges)
        S2_conv = self.C["l4"]["d2"](L[2], S2.transpose(1, 0))
        S2_conv = self.activaction(S2_conv)

        S2_concat = torch.cat((S1_lifted, S2_conv.transpose(1, 0)), dim=1)

        S2 = self.aggr["l4"]["d2"](S2_concat)

        ###### layer 5 ######

        # S0 = conv(S0)
        S0 = self.C["l4"]["d0"](L[0], S0)
        S0 = self.activaction(S0)

        # (num_edges, num_filters x c_in)
        S0_lifted = self.lift(Bts[0], S0)

        # (num_filters x c_in, num_edges)
        S1_conv = self.C["l4"]["d1"](L[1], S1.transpose(1, 0))
        S1_conv = self.activaction(S1_conv)

        S1_concat = torch.cat((S0_lifted, S1_conv.transpose(1, 0)), dim=1)

        S1 = self.aggr["l5"]["d1"](S1_concat)

        # (num_edges, num_filters x c_in)
        S1_lifted = Bts[1] @ S1

        # (num_filters x c_in, num_edges)
        S2_conv = self.C["l4"]["d2"](L[2], S2.transpose(1, 0))

        S2_concat = torch.cat((S1_lifted, S2_conv.transpose(1, 0)), dim=1)

        S2 = self.last_aggregator(S2_concat)

        return S2

    def lift(self, B, S):
        return B @ S.transpose(1, 0)

    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """

        triangle_normals = batch["triangle_normals"][0]

        preds = self.get_predictions(batch)

        criterion = nn.MSELoss(reduction="mean")
        loss = torch.tensor(0.0).type_as(preds)
        for i in range(3):
            loss += criterion(preds[:, i], triangle_normals[:, i])

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        preds = self.get_predictions(batch)
        # TODO: fix batching
        targets = batch["triangle_normals"][0]
        return {"preds": preds, "targets": targets}

    def test_epoch_end(self, test_batch_outputs):
        preds = torch.cat([batch["preds"] for batch in test_batch_outputs], dim=0)
        targets = torch.cat([batch["targets"] for batch in test_batch_outputs], dim=0)

        per_coord_diffs = compute_per_coord_diff(preds, targets)
        normalized_preds = normalize(preds, dim=1)
        angle_diff = compute_angle_diff(normalized_preds, targets)

        self.log_dict(
            {
                "x_diff": per_coord_diffs[0],
                "y_diff": per_coord_diffs[1],
                "z_diff": per_coord_diffs[2],
                "angle_diff": angle_diff,
            }
        )

        self.plotter.pred_normals = normalized_preds
        self.plot_results()

    def get_predictions(self, batch):
        node_positions, triangle_normals = (
            batch["node_positions"][0],
            batch["triangle_normals"][0],
        )
        boundaries = [batch["B1"][0], batch["B2"][0]]
        laplacians = [
            batch["node_laplacian"][0],
            batch["edge_laplacian"][0],
            batch["triangle_laplacian"][0],
        ]

        preds = self(X=node_positions, boundaries=boundaries, laplacians=laplacians)
        return preds

    def plot_results(self):
        target_norm_colors = self.plotter.transform_normals_to_rgb()
        true_plot = self.plotter.plot_mesh(
            title="True normals", colors=target_norm_colors
        )
        true_plot.show()

        predicted_norm_colors = self.plotter.transform_normals_to_rgb(pred=True)
        pred_plot = self.plotter.plot_mesh(
            title="Predicted normals", colors=predicted_norm_colors
        )
        pred_plot.show()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
