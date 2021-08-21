# deep learning
import torch

# data
import pandas as pd
import numpy as np

# ad hoc
from inter_order.utils.misc import *
from inter_order.models.mesh_scnn import MySCNN
from inter_order.data.datamodule import NormalsDataModule
from pytorch_lightning import Trainer

# misc
import os

print_torch_version()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_folder = "data/inter_order"

# run settings
USE_NOISY = True
mesh_name = "bob"

# data loading

to_load = {
    "laplacians": list,
    "boundaries": list,
    "positions": dict,
    "noisy_positions": dict,
    "original_positions": list,
    "normals": dict,
    "triangles": list,
}

load_folder = os.path.join(data_folder, mesh_name)
loaded_data = load_data(to_load, load_folder)

data_module = NormalsDataModule(loaded_data)

filter_size = 30

# model = LinearBaseline(num_nodes, num_triangles).to(device)
# model = TopologyAwareBaseline(num_nodes, num_triangles, node_triangle_adj).to(device)
model = MySCNN(filter_size, colors=3).to(device)

num_epochs = 10
trainer = Trainer(max_epochs=num_epochs, gpus=1)
trainer.fit(model, data_module)

trainer.test(model)
