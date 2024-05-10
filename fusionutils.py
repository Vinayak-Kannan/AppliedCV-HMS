import albumentations as A
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import datetime
import time
import joblib
import torch
import torch.nn as nn
import random
import timm
import librosa
import copy
import csv
import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from torchtest import assert_vars_change
#DO NOT DELETE THIS
from xgboost import XGBClassifier


USE_WAVELET = None

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]



def seed_everything(seed: int):
    #set random seed in all packages for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def spectrogram_from_eeg(parquet_path, display=False):
    '''converts EEG series to spectrogram'''

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')

    if display:
        plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # # DENOISE
            # if USE_WAVELET:
            #     x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256,
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40
            img[:,:,k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0

        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            # plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')

    if display:
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        # plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print(); print('#'*25); print()

    return img

class FusionDataset(Dataset):
    ''' 
    Custom dataset for complex fusion model input.
    Uses EN, Rocket, and XGBoost models to generate data
    '''
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        rocket_model,
        xgboost_model,
        paths
    ):
        self.df = df
        self.config = config
        self.rocket_model = rocket_model
        self.xgboost_model = xgboost_model
        self.paths = paths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            en_X, y = self.__EN_data_generation(row)
            rocket_in = self.get_rocket_output(row)
            feature_row = pd.read_csv(self.paths.FINAL_FEATURE_FOLDER + str(row.eeg_id) + '.csv').iloc[:, 2:]
        except FileNotFoundError as e:
          return None
        xg_in = self.get_xgboost_output(feature_row)
        return (torch.tensor(en_X, dtype=torch.float32),
                rocket_in,
                xg_in,
                torch.tensor(y, dtype=torch.float32))

    def get_rocket_output(self,row):
        pq = pd.read_parquet(f"{self.paths.TRAIN_EEGS}{row.eeg_id}.parquet")
        middle = (len(pq)-2_000)//2
        pq = pq.iloc[middle:middle+2_000:2]
        #just doing this so the formatting works well
        pq = [pq.reset_index()]

        batch_size = 1
        x_batch = pd.concat(pq,keys=list(range(batch_size)),axis=0).reset_index(level=1)
        x_batch['instances'] = x_batch.index
        x_batch = x_batch.rename(columns={"level_1": "timepoints"})
        x_batch = x_batch.set_index(['instances', 'timepoints'])
        x_batch = x_batch.fillna(0)
        rocket_predictions = self.rocket_model.predict(x_batch)
        rocket_predictions = torch.from_numpy(rocket_predictions.to_numpy())
        return rocket_predictions  #, self.df.iloc[ind]['expert_consensus']

    def get_xgboost_output(self,feature_row):
        x = feature_row.values.reshape(1, -1)
        xg_probs = self.xgboost_model.predict_proba(x)[0]
        return torch.tensor(xg_probs, dtype=torch.float32)
    
    def __EN_data_generation(self, row):
        """
        Generates usable data from parquets
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        r = 0
        spectrogram_file_path = self.paths.TRAIN_SPECTROGRAMS + str(row.spectrogram_id) + ".parquet"
        spectrogram = pd.read_parquet(spectrogram_file_path).iloc[:,1:].values
        eeg_file_path = self.paths.TRAIN_EEGS + str(row.eeg_id) + ".parquet"
        eeg = spectrogram_from_eeg(eeg_file_path)

        for region in range(4):
            img = spectrogram[r:r+300, region*100:(region+1)*100].T

            # Log transform spectrogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img-mu)/(std+ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = eeg
            X[:, :, 4:] = img

        y = row[self.config.LABEL_COLS].values.astype(np.float32)
        return X, y


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    X = torch.stack([item[0] for item in batch])
    rocket = torch.stack([item[1] for item in batch])
    xg = torch.stack([item[2] for item in batch])
    y = torch.stack([item[3] for item in batch])

    return {'X': X, 'rocket': rocket, 'xg': xg, 'label': y }

class EfficientNet(nn.Module):
    '''
    Custom EfficientNet model for EEG data
    '''
    def __init__(self, config, num_classes: int = 6):
        super(EfficientNet, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=False
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )

        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten())

        self.featurizer = True

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # === Get spectrograms ===
        spectrograms = [x[:, :, :, i:i+1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # === Get EEG spectrograms ===
        eegs = [x[:, :, :, i:i+1] for i in range(4,8)]
        eegs = torch.cat(eegs, dim=1)

        # === Reshape (512,512,3) ===
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectrograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectrograms

        x = torch.cat([x,x,x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        if self.featurizer:
            x = self.adapter(x)
        else:
            x = self.custom_layers(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, EN_Model, freeze = False):
        super(FusionModel, self).__init__()
        self.EfficientNet = EN_Model
        self.EN_out = 1280
        self.encoder_out = 48
        self.hidden = 10
        self.num_classes = 6
        if not freeze:
            for param in self.EfficientNet.parameters():
                param.requires_grad = True

        self.encoder = nn.Linear(self.EN_out, self.encoder_out)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_out + 6 + 6, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.num_classes)
        )

    def forward(self, X, rocket, xg, label=None):
        EN_features = self.EfficientNet(X)
        EN_features = F.relu(self.encoder(EN_features))
        try:
            combined_features = torch.cat((EN_features, rocket.squeeze(), xg), dim=1)
        except:
            # print(EN_features, rocket.squeeze(), xg)
            combined_features = torch.cat((EN_features, rocket.squeeze().unsqueeze(0), xg), dim=1)
            # raise
        out = self.classifier(combined_features)
        return F.softmax(out, dim = 1)

def train_model(model, criterion, optimizer, dataloaders, device, paths, config, num_epochs=5,):
    torch.backends.cudnn.enabled = False
    since = time.time()
    best_acc = 0.0
    current = datetime.datetime.now().minute
    if isinstance(criterion, nn.KLDivLoss):
        print("KL")
        KL = True
    else:
        KL = False

    dataset_sizes  = {x: len(dataloaders[x]) * config.BATCH_SIZE for x in ['train', 'val']}

    with open(f'/content/gcs/fusion/fusion_training_results{str(current)}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'phase', 'loss', 'accuracy'])
        print('result file: ' + f'/content/gcs/fusion/fusion_training_results{str(current)}.csv')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for _,batch in tqdm(enumerate(dataloaders[phase], 0), unit="batch", total=len(dataloaders[phase])):
                    X = batch['X'].to(device)
                    rocket = batch['rocket'].to(device)
                    xg = batch['xg'].to(device)
                    labels = batch['label'].to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(X, rocket, xg)
                    
                        _, preds = torch.max(outputs, 1)
                        _, consensus = torch.max(labels, 1)
                        if not KL:
                            loss = criterion(outputs, labels/torch.sum(labels, dim = 1).unsqueeze(1))
                        else:
                            loss = criterion(torch.log(outputs),labels/torch.sum(labels, dim = 1).unsqueeze(1))
                        print("loss: ",end = '')
                        print('{:0.3f}'.format(loss.item()))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                   # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == consensus.data)


                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print()
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                writer.writerow([epoch, phase, epoch_loss, epoch_acc])

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    print('saving')
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), paths.SAVE_PATH + f'fusion_epoch_{epoch}.pth')

            print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    return model

def test_model(model, criterion, testloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    dataset_size = len(testloader.dataset)

    with torch.no_grad():
        for batch in tqdm(testloader, unit="batch", total=len(testloader)):
            X = batch['X'].to(device)
            rocket = batch['rocket'].to(device)
            xg = batch['xg'].to(device)
            labels = batch['label'].to(device)

            outputs = model(X, rocket, xg)
            if isinstance(criterion, nn.KLDivLoss):
                loss = criterion(torch.log(outputs), labels/torch.sum(labels, dim=1).unsqueeze(1))
            else:
                loss = criterion(outputs, labels/torch.sum(labels, dim=1).unsqueeze(1))
            
            _, preds = torch.max(outputs, 1)
            _, consensus = torch.max(labels, 1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == consensus.data)

    average_loss = running_loss / dataset_size
    accuracy = running_corrects.double() / dataset_size
    print(f'Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
    return average_loss, accuracy