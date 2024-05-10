import albumentations as A
import gc
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
from sktime.utils import mlflow_sktime
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from typing import Dict, List
from glob import glob

from fusionutils import *
from fusionconfig import *

def main():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    paths.prepend_path_prefix("/home/Ramizire")
    seed_everything(config.SEED)

    fusion_data = pd.read_csv(paths.FUSION_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rocket_model = mlflow_sktime.load_model(model_uri=paths.ROCKET_DIR)
    xg_model = joblib.load(paths.XG_MODEL)

    fusion_dataset = FusionDataset(fusion_data, config,rocket_model, xg_model, paths)

    def worker_init_fn(worker_id):
        dataset = torch.utils.data.get_worker_info().dataset
        dataset.xgboost_model = joblib.load(paths.XG_MODEL)

    excess, fusion_dataset = random_split(fusion_dataset, [config.EXCESS, 1-config.EXCESS], generator=torch.Generator().manual_seed(0))
    train_set, val_set = random_split(fusion_dataset, [.8, .2], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn = collate_fn,
        # worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn = collate_fn,
        # worker_init_fn=worker_init_fn
    )


    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train':len(train_loader), 'val': len(val_loader)}

    EN_model = EfficientNet(config)
    checkpoint = torch.load(paths.MODEL_WEIGHTS)
    EN_model.load_state_dict(checkpoint["model"])
    EN_model.to(device)

    Fusion_Model = FusionModel(EN_model, freeze=True).to(device)

    optimizer = torch.optim.Adam(Fusion_Model.parameters(), lr=config.LR)
    criterion = nn.KLDivLoss(reduction = 'batchmean')
    # criterion = nn.CrossEntropyLoss()

    model = train_model(Fusion_Model, criterion, optimizer,dataloaders,device, paths, config, num_epochs=config.N_EPOCHS )

if __name__ == '__main__':
    main()