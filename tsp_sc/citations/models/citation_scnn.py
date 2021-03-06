import pytorch_lightning as pl
import wandb

from tsp_sc.common.misc import similar
import torch
import torch.nn as nn
import pandas as pd


class CitationSCNN(pl.LightningModule):
    def __init__(self, params):
        self.name = params["name"]
        super().__init__()

    def training_step(self, batch, batch_idx):

        targets = [batch[f"Y{d}"] for d in range(self.num_dims)]

        train_indices = [batch[f"train_indices_{d}"] for d in range(self.num_dims)]

        preds = self.get_preds(batch)

        considered_simplex_dim = len(targets) - 1

        criterion = nn.L1Loss(reduction="mean")
        loss = torch.FloatTensor([0.0]).type_as(targets[0])

        for k in range(0, considered_simplex_dim + 1):
            # compute the loss over the k-th dimension of the sample b (0 unless batch) over the known simplices
            dim_k_loss = criterion(
                preds[k][0, train_indices[k]], targets[k][0, train_indices[k]]
            )
            self.log(f"train/loss_dim_{k}", dim_k_loss, on_epoch=True, logger=True)
            loss += dim_k_loss

        self.log("train/loss", loss, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        targets = [batch[f"Y{d}"] for d in range(self.num_dims)]

        val_indices = [batch[f"val_indices_{d}"] for d in range(self.num_dims)]

        preds = self.get_preds(batch)

        considered_simplex_dim = len(targets) - 1

        criterion = nn.L1Loss(reduction="mean")
        val_loss = torch.FloatTensor([0.0]).type_as(targets[0])

        for k in range(0, considered_simplex_dim + 1):
            # compute the loss over the k-th dimension of the sample b (0 unless batch) over the known simplices
            dim_k_loss = criterion(
                preds[k][0, val_indices[k]], targets[k][0, val_indices[k]]
            )
            val_loss += dim_k_loss
            self.log(f"val/loss_dim_{k}", dim_k_loss, on_epoch=True, logger=True)
            val_diff = preds[k][0, val_indices[k]] - targets[k][0, val_indices[k]]
            self.trainer.logger.experiment.log(
                {f"val_diff/diff_dim_{k}": wandb.Histogram(val_diff.cpu())}
            )

        self.log("val/loss", val_loss, on_epoch=True, logger=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        preds = self.get_preds(batch)

        targets = [batch[f"Y{d}"] for d in range(self.num_dims)]
        test_indices = [batch[f"test_indices_{d}"] for d in range(self.num_dims)]

        return {"preds": preds, "targets": targets, "test_indices": test_indices}

    def test_epoch_end(self, test_batch_outputs):
        preds = [batch["preds"] for batch in test_batch_outputs][0]
        targets = [batch["targets"] for batch in test_batch_outputs][0]
        test_indices = [batch["test_indices"] for batch in test_batch_outputs][0]

        criterion = nn.L1Loss(reduction="mean")
        test_loss = torch.FloatTensor([0.0]).type_as(targets[0])

        for k in range(0, self.num_dims):
            # compute the loss over the k-th dimension of the sample b (0 unless batch) over the known simplices
            dim_k_loss = criterion(
                preds[k][0, test_indices[k]], targets[k][0, test_indices[k]]
            )
            test_loss += dim_k_loss
            self.log(f"test/loss_dim_{k}", dim_k_loss, logger=True)

        self.log("test/loss", test_loss, logger=True)

        margins = [0.3]
        only_missing_simplices = True

        accuracies = self.compute_accuracy_margins(
            preds, targets, margins, test_indices, only_missing_simplices
        )
        accuracy_report = self.get_accuracy_report(accuracies)
        self.trainer.logger.experiment.log(
            {f"Accuracy": pd.DataFrame(accuracy_report, index=[0])}
        )
        self.log_dict(accuracy_report)

    def get_components_from_batch(self, batch):
        """
        Num_complexes is currently 1, therefore we return the first
        :param batch:
        :return:
        """

        L = [batch[f"L{d}"][0] for d in range(self.num_dims)]
        I = [batch[f"I{d}"][0] for d in range(self.num_dims)]
        S = [None if d == 0 else batch[f"S{d}"][0] for d in range(self.num_dims)]

        components = {"full": L, "irr": I, "sol": S}

        return components

    def compute_accuracy_margins(
        self, preds, targets, margins, known_indices, only_missing_simplices
    ):
        accuracies = {margin: [] for margin in margins}

        for margin in margins:
            margin_accuracies = self.compute_accuracy_margin(
                preds, targets, margin, known_indices, only_missing_simplices
            )
            accuracies[margin].append(margin_accuracies)

        return accuracies

    def compute_accuracy_margin(
        self, preds, targets, margin, test_indices, only_missing_simplices
    ):
        """
        returns the accuracy for each dimension by counting the number
        of hits over the total number of simplices of that dimension
        if only_missing_simplices is True, then the accuracy is computed only over the missing simplices
        """
        test_indices_set = [
            set([tens.item() for tens in test_indices[i]])
            for i in range(len(test_indices))
        ]

        dims = len(targets)
        accuracies = []

        for k in range(dims):

            hits = 0
            den = 0

            (_, num_simplices_dim_k) = preds[k].shape

            for j in range(num_simplices_dim_k):

                # if we only compute the accuracy over the missing simplices,
                # then we skip this simplex if it is known
                if only_missing_simplices:
                    if j not in test_indices_set[k]:
                        continue

                curr_value_pred = preds[k][0][j]
                curr_value_true = targets[k][0][j]
                den += 1

                if similar(curr_value_pred, curr_value_true, margin):
                    hits += 1

            accuracy = round(hits / den, 4)
            accuracies.append(accuracy)

        return accuracies

    def get_accuracy_report(self, accuracies):
        accuracy_report = {}
        for margin, margin_accuracies in accuracies.items():
            margin_accuracies = margin_accuracies[0]
            for dim, acc in enumerate(margin_accuracies):
                accuracy_report[f"margin-{margin}-dim-{dim}"] = acc
        return accuracy_report
