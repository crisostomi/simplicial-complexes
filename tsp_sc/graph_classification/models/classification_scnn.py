import torch.nn as nn
from tsp_sc.common.simplicial_convolution import MySimplicialConvolution
from tsp_sc.common.simp_complex import ComplexBatch
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from tsp_sc.common.utils import get_pooling_fn, get_nonlinearity
from torchmetrics import F1, Precision, Recall, Accuracy


class ClassificationSCNN(pl.LightningModule):
    def __init__(self, params):
        """
        parameters:
            filter_size: size of the convolutional filter
            colors: number of channels
            component_to_use: whether to use both components or one of the two, values can be 'both', 'sol' or 'irr'
    """
        super().__init__()

        self.F1 = F1(average="micro")
        self.prec = Precision(average="micro")
        self.recall = Recall(average="micro")
        self.accuracy = Accuracy()

        self.colors = params["colors"]
        self.aggregation = params["aggregation"]
        self.component_to_use = params["component_to_use"]
        self.keep_separated = params["keep_separated"]
        self.filter_size = params["filter_size"]
        self.readout = params["readout"]
        self.global_nonlinearity = params["global_nonlinearity"]
        self.hidden_size = params["hidden_size"]
        self.num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]
        self.feature_dim = params["num_features"]

        num_filters = 30
        variance = 0.01
        self.num_layers = 3
        self.num_dims = 3
        self.dims = range(0, self.num_dims)
        self.comps = [["irr"] if dim == 0 else ["sol", "irr"] for dim in self.dims]
        self.layers = range(0, self.num_layers)

        self.C = nn.ModuleDict({f"d{j}": nn.ModuleDict() for j in self.dims})

        for dim in self.dims:

            self.C[f"d{dim}"] = nn.ModuleList()

            first_conv = nn.ModuleDict(
                {
                    comp: MySimplicialConvolution(
                        self.filter_size,
                        C_in=self.feature_dim[dim],
                        C_out=self.hidden_size,
                        variance=variance,
                    )
                    for comp in self.comps[dim]
                }
            )
            self.C[f"d{dim}"].append(first_conv)

            for layer in range(1, self.num_layers):
                convs = nn.ModuleDict(
                    {
                        comp: MySimplicialConvolution(
                            self.filter_size,
                            C_in=self.hidden_size,
                            C_out=self.hidden_size,
                            variance=variance,
                        )
                        for comp in self.comps[dim]
                    }
                )
                self.C[f"d{dim}"].append(convs)

        self.L = nn.ModuleDict(
            {
                f"d{dim}": nn.ModuleList(
                    [
                        nn.Linear(2 * self.hidden_size, self.hidden_size,)
                        for i in self.layers
                    ]
                )
                for dim in self.dims[1:]
            }
        )
        self.final_lin1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.final_lin2 = nn.Linear(self.hidden_size, self.num_classes)

        self.pooling_fn = get_pooling_fn(self.readout)

    def forward(self, inputs, components, batch):
        """
        parameters:
            components: dict of lists, keys: 'full' for the Laplacian, 'sol' for the solenoidal and 'irr' for irrotational component
                        for each component, there is a list of length self.dims containing for each dimension the component for that dimension
            xs: inputs
        """

        layers = range(self.num_layers)
        last_layer = self.num_layers - 1

        dims = range(batch.dimension + 1)

        comps = self.get_comps(batch.dimension)

        activ = {layer: nn.LeakyReLU() for layer in layers}

        outs = [{} for layer in layers]

        for layer in layers:
            for dim in dims:
                prev_output = (
                    inputs[dim]
                    if layer == 0
                    else activ[layer](outs[layer - 1][f"d{dim}"])
                )
                comp_outputs = [
                    self.convolve(prev_output, components, layer, dim, comp)
                    for comp in comps[f"d{dim}"]
                ]
                aggregated = self.aggregate(comp_outputs, layer, dim)

                outs[layer][f"d{dim}"] = aggregated

        cochain_outputs = [
            outs[last_layer][f"d{i}"] for i in range(batch.dimension + 1)
        ]

        pooled_xs = self.pool_complex(cochain_outputs, batch)
        x = pooled_xs.sum(dim=0)

        model_nonlinearity = get_nonlinearity(
            self.global_nonlinearity, return_module=False
        )

        x = model_nonlinearity(self.final_lin1(x))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final_lin2(x)

        return x

    @staticmethod
    def get_comps(dim):
        comps = {}
        dims = range(dim + 1)
        for dim in dims:
            if dim == 0:
                dim_comps = ["irr"]
            elif dim == dim:
                dim_comps = ["sol"]
            else:
                dim_comps = ["sol", "irr"]
            comps[f"d{dim}"] = dim_comps
        return comps

    def convolve(self, input, components, layer, dim, component):
        """
        Convolves input using the given component

        parameters:
            input: input signal to convolve
            components: dict of lists, keys: 'full' for the Laplacian, 'sol' for the solenoidal and 'irr' for irrotational component
                        for each component, there is a list of length self.dims containing for each dimension the component for that dimension
            layer: int, number of the layer
            dim: int, dimension of the simplices being considered
            component: string, 'full', 'sol' or 'irr'
    """
        convolution = self.C[f"d{dim}"][layer][component]
        # print(dim, component, layer)
        return convolution(components[component][dim], input)

    def merge_components(self, outs):
        """
        merges the outputs of the final layer
        if self.component_to_use is 'sol' or 'irr', then it only uses that component
        otherwise for each dimension the components are aggregated
    """

        if self.component_to_use != "both":
            final_out1, final_out2 = (
                outs["l3"]["d1"][self.component_to_use],
                outs["l3"]["d2"][self.component_to_use],
            )

        else:
            final_out1 = self.aggregate(
                outs["l3"]["d1"]["sol"], outs["l3"]["d1"]["irr"], "d1"
            )
            final_out2 = self.aggregate(
                outs["l3"]["d2"]["sol"], outs["l3"]["d2"]["irr"], "d2"
            )

        return final_out1, final_out2

    def aggregate(self, components_outputs, layer, dim):
        """
        Aggregates the output of the convolution over the different components
        if the output is from a single component (e.g. from the irrotational components for nodes), then it just returns it
        otherwise, depending on self.aggregation either aggregates by summing or by using a MLP
        """

        if len(components_outputs) == 1:
            return components_outputs[0]

        # aggregation via MLP
        else:

            (c_out, num_simplices) = components_outputs[0].shape

            reshaped_outs = [
                out.reshape(num_simplices, c_out) for out in components_outputs
            ]

            out_concat = torch.cat(reshaped_outs, 1)

            out = self.L[f"d{dim}"][layer](out_concat)

            out = out.reshape(c_out, num_simplices)

        return out

    def training_step(self, batch: ComplexBatch, batch_idx):

        preds = self.get_preds(batch)

        labels = batch.get_labels()

        loss = nn.CrossEntropyLoss()(preds, labels)

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: ComplexBatch, batch_idx):
        preds = self.get_preds(batch)

        labels = batch.get_labels()

        loss = nn.CrossEntropyLoss()(preds, labels)

        self.log("val/loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        preds = torch.argmax(preds, -1)
        targets = batch.get_labels()

        acc = self.accuracy(preds, targets)

        self.log("val/acc_step", acc, on_epoch=True, logger=True)

        return {"val/loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, val_batch_outputs):
        preds = [batch["preds"] for batch in val_batch_outputs]
        targets = [batch["targets"] for batch in val_batch_outputs]

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        acc = self.accuracy(preds, targets)

        self.log("val/acc_epoch", acc, on_epoch=True, logger=True)

    def get_preds(self, batch):

        inputs = [
            batch.cochains[i]["signal"].transpose(1, 0)
            for i in range(batch.dimension + 1)
        ]

        components = self.get_components_from_batch(batch)

        preds = self(inputs, components, batch)
        return preds

    def get_components_from_batch(self, batch):
        dim = batch.dimension
        cochains = batch.cochains

        S = [None] + [cochains[i]["solenoidal"] for i in range(1, dim + 1)]
        I = [cochains[i]["irrotational"] for i in range(dim)] + [None]
        L = [cochains[i]["laplacian"] for i in range(dim + 1)]

        components = {"full": L, "irr": I, "sol": L}
        return components

    def test_step(self, batch, batch_idx):
        preds = self.get_preds(batch)
        preds = torch.argmax(preds, -1)
        targets = batch.get_labels()

        return {"preds": preds, "targets": targets}

    def test_epoch_end(self, test_batch_outputs):
        preds = [batch["preds"] for batch in test_batch_outputs]
        targets = [batch["targets"] for batch in test_batch_outputs]

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        F1 = self.F1(preds, targets)
        prec = self.prec(preds, targets)
        recall = self.recall(preds, targets)
        acc = self.accuracy(preds, targets)

        self.log("test/F1", F1, on_epoch=True, logger=True)
        self.log("test/recall", prec, on_epoch=True, logger=True)
        self.log("test/prec", recall, on_epoch=True, logger=True)
        self.log("test/acc", acc, on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def pool_complex(self, xs, batch):

        xs = [x.transpose(1, 0) for x in xs]
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = batch.cochains[0].batch.max() + 1

        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(
            self.num_dims, batch_size, xs[0].size(-1), device=batch_size.device
        )
        for i in range(len(xs)):
            pooled = self.pooling_fn(xs[i], batch.cochains[i].batch, size=batch_size)
            pooled_xs[i, :, :] = pooled
        return pooled_xs
