import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
# import multiprocessing
import numpy as np
import os
import pandas as pd
import pywt
import random
import time
import torch
import torch.nn as nn
import random
import timm
import librosa
import joblib
import warnings
import copy
import csv
import tqdm
from sktime.utils import mlflow_sktime
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm

from typing import Dict, List
from xgboost import XGBClassifier
from albumentations.pytorch import ToTensorV2
from glob import glob

from fusionutils import *
from fusionconfig import *

def main():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    paths.prepend_path_prefix("/home/Ramizire")
    seed_everything(0)

    fusion_data = pd.read_csv(paths.FUSION_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rocket_model = mlflow_sktime.load_model(model_uri=paths.ROCKET_DIR)
    xg_model = joblib.load(paths.XG_MODEL)

    # feature_row = pd.read_csv(paths.FINAL_FEATURE_FOLDER + str(582999) + '.csv').iloc[:, 2:].values.reshape(1, -1)
    # xg_probs = xg_model.predict_proba(feature_row)[0]
    # print(feature_row[0])
    # print(xg_probs)
    # 1/0
    fusion_dataset = FusionDataset(fusion_data, config,rocket_model, xg_model, paths)

    excess, fusion_dataset = random_split(fusion_dataset, [config.EXCESS, 1-config.EXCESS], generator=torch.Generator().manual_seed(0))
    train_set, val_set = random_split(fusion_dataset, [.7, .3], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn = collate_fn
    )
    val_loader = DataLoader(val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn = collate_fn
    )


    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train':len(train_loader), 'val': len(val_loader)}

    EN_model = EfficientNet(config)
    checkpoint = torch.load(paths.MODEL_WEIGHTS)
    EN_model.load_state_dict(checkpoint["model"])
    EN_model.to(device)

    Fusion_Model = FusionModel(EN_model, freeze=True).to(device)

    optimizer = torch.optim.Adam(Fusion_Model.parameters(), lr=config.LR)
    # criterion = nn.KLDivLoss(reduction = 'batchmean')
    criterion = nn.CrossEntropyLoss()

    model = train_model(Fusion_Model, criterion, optimizer,dataloaders,device, paths, config, num_epochs=config.N_EPOCHS )

if __name__ == '__main__':
    main()