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
    paths.prepend_path_prefix("/home/Ramizire")
    seed_everything(config.SEED)

    #data loading 
    test_df = pd.read_csv(paths.TEST_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rocket_model = mlflow_sktime.load_model(model_uri=paths.ROCKET_DIR)
    xg_model = joblib.load(paths.XG_MODEL)
    testset = FusionDataset(test_df, config,rocket_model, xg_model, paths)
    testloader = DataLoader(testset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn = collate_fn,
    )
    
    #model loading 
    EN_model = EfficientNet(config)
    Fusion_Model = FusionModel(EN_model, freeze=True).to(device)
    Fusion_Model.to(device)
    print('loading')
    weights = torch.load('/home/Ramizire/content/gcs/models/fusion_epoch_7.pth')
    Fusion_Model.load_state_dict(weights)
    criterion = nn.KLDivLoss(reduction = 'batchmean')

    test_model(Fusion_Model, criterion, testloader, device)


if __name__ == '__main__':
    main()