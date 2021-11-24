import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

import torch_sparse
from tsp_sc.common.io import *
from tsp_sc.common.misc import *
from tsp_sc.graph_classification.utils import get_paths
from tsp_sc.graph_classification.data.datamodule import GraphClassificationDataModule
from tsp_sc.graph_classification.models.classification_scnn import ClassificationSCNN
from tsp_sc.graph_classification.data.datasets.tu import TUDataset
from pytorch_lightning import Trainer
from tsp_sc.graph_classification.data.data_loading import load_dataset
from tsp_sc.graph_classification.data.dataloader import DataLoader
from tsp_sc.graph_classification.utils import get_model
import wandb
from tsp_sc.citations.utils.utils import fix_dict_in_config

parser = argparse.ArgumentParser()
parser.add_argument("config")
cli_args = parser.parse_args()

wandb.login()

config = load_config(cli_args.config)

path_params, data_params, run_params, model_params = (
    config["paths"],
    config["data"],
    config["run_params"],
    config["models"],
)

assert data_params["considered_simplex_dim"] <= data_params["max_simplex_dim"]

device = "cuda" if torch.cuda.is_available else "cpu"
paths = get_paths(path_params, data_params)

name = "PROTEINS"
root = paths["data"]
dataset = load_dataset(name, root, max_dim=2, init_method="sum")

dataloader = DataLoader(dataset, batch_size=2)

model_name = "classification_scnn"

wandb.init(config=config)
fix_dict_in_config(wandb)

model_params = wandb.config["models"][model_name]

model = get_model(model_name, model_params)

early_stopping_callback = EarlyStopping(
    monitor="val/loss",
    min_delta=run_params["min_delta"],
    patience=run_params["patience"],
    verbose=False,
    mode="min",
)

run_config = get_run_config(model_name, config)
wandb_logger = WandbLogger(
    name=model_name, project="complex-classification", config=run_config
)

trainer = Trainer(
    max_epochs=run_params["max_epochs"],
    gpus=run_params["gpus"],
    logger=wandb_logger,
    callbacks=[early_stopping_callback],
)

data_module = GraphClassificationDataModule(dataloader)
trainer.fit(model, data_module)

# trainer.test(model)

wandb.finish()
