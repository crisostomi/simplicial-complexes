import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from tsp_sc.common.io import *
from tsp_sc.common.misc import *
from tsp_sc.graph_classification.utils import get_paths
from tsp_sc.graph_classification.data.datamodule import GraphClassificationDataModule
from pytorch_lightning import Trainer
from tsp_sc.graph_classification.utils import get_model, load_dataset
import wandb
from tsp_sc.citations.utils.utils import fix_dict_in_config

parser = argparse.ArgumentParser()
parser.add_argument("--config")
cli_args = parser.parse_args()

wandb.login()

config = load_config(cli_args.config)

wandb.init(config=config)

fix_dict_in_config(wandb)

path_params, data_params, run_params, model_params = (
    wandb.config["paths"],
    wandb.config["data"],
    wandb.config["run_params"],
    wandb.config["models"],
)

assert data_params["considered_simplex_dim"] <= data_params["max_simplex_dim"]

device = "cuda" if torch.cuda.is_available else "cpu"
paths = get_paths(path_params, data_params)

data_module = GraphClassificationDataModule(paths, data_params)

model_name = "classification_scnn"


model_params = wandb.config["models"][model_name]
model_params["num_classes"] = data_module.dataset.num_classes
model_params["num_features"] = data_module.dataset.num_features()

model = get_model(model_name, model_params)

early_stopping_callback = EarlyStopping(
    monitor="val/loss",
    min_delta=run_params["min_delta"],
    patience=run_params["patience"],
    verbose=False,
    mode="min",
)

run_config = get_run_config(model_name, wandb.config)

wandb_logger = WandbLogger(name=model_name, project="TSP-SC", config=run_config)
wandb.define_metric("val/acc", summary="max", goal="max", step_metric="epoch")
wandb.define_metric("val/loss", summary="min", goal="min", step_metric="epoch")

trainer = Trainer(
    max_epochs=run_params["max_epochs"],
    min_epochs=run_params["min_epochs"],
    gpus=run_params["gpus"],
    logger=wandb_logger,
    callbacks=[early_stopping_callback],
    log_every_n_steps=1,
)


trainer.fit(model, data_module)

if data_module.test_dataloader() is not None:
    trainer.test(model, test_dataloaders=data_module)

wandb.finish()
