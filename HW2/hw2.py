import gc
import os
from time import time
import random
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


##### Hyper-parameters #######################################################
# data prarameters
# the number of frames to concat with, n must be odd (total 2k+1 = n frames)
concat_nframes = 21
# the ratio of data used for training, the rest will be used for validation
train_ratio = 0.8

# training parameters
seed = 0                        # random seed
batch_size = 256                # batch size
num_epochs = 50                   # the number of training epoch
learning_rate = 1e-3          # learning rate
# the path where the checkpoint will be saved
model_path = './HW2/models/model.ckpt'

# model parameters
# the input dim of the model, you should not change the value
input_dim = 39 * concat_nframes
hidden_layers = 6               # the number of hidden layers
hidden_dim = 2048                # the hidden dim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')


##### Functions ##############################################################
# fix seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# data
def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(
        1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41  # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(
            phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(
            phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(
            phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError(
            'Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) +
          ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list), position=0, leave=False):
        feat = torch.load(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
            y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

# model
class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim)
              for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

##### Prepare dataset and model ##############################################
same_seeds(seed)
# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./HW2/datasets/feat',
                                   phone_path='./HW2/datasets', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./HW2/datasets/feat',
                               phone_path='./HW2/datasets', concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# get dataloader
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers,
                   hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, num_epochs, eta_min=0)


##### Training ###############################################################
best_acc = 0.0
not_imporving = 0
start_time = time()
for epoch in range(num_epochs):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader, position=0, leave=False)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # get the index of the class with the highest probability
        _, train_pred = torch.max(outputs, 1)
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    scheduler.step()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, position=0, leave=False)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                # get the index of the class with the highest probability
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f} | time: {:4.2f}'.format(
                epoch + 1, num_epochs, train_acc/len(train_set), train_loss/len(
                    train_loader), val_acc/len(val_set), val_loss/len(val_loader), time() - start_time
            ))

            # if the model improves, save a checkpoint at this epoch. If the model is not improving for 5 epochs, stops the training.
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(
                    best_acc/len(val_set)))
                not_imporving = 0
            else:
                not_imporving += 1
                if not_imporving >= 5:
                    print(
                        f"epoch {epoch}: model not imporving. stop training...")
                    break

    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epochs, train_acc /
            len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
del train_loader, val_loader
gc.collect()


##### Testing ################################################################
# load data
test_X = preprocess_data(split='test', feat_dir='./HW2/datasets/feat',
                         phone_path='./HW2/datasets', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers,
                   hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

# Make prediction.
test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, position=0, leave=False)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        # get the index of the class with the highest probability
        _, test_pred = torch.max(outputs, 1)
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# Write prediction to a CSV file.
with open('./HW2/prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))