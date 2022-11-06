from torch import nn
import os
import random
from time import time
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, SubsetRandomSampler
from torchvision.datasets import DatasetFolder, VisionDataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


##### Hyper-parameters #######################################################
# training parameters
ResNET_path = './HW3/models/ResNET.ckpt'
VGG11_path = './HW3/models/VGG11.ckpt'
dataset_path = './HW3/datasets'
seed = 0                        # random seed

batch_size = 96                # batch size
num_epochs = 10  # (10*4)*10                 # the number of training epoch

learning_rate = 1e-3          # learning rate
weight_decay_value = 1e-5
patience = 15  # If no improvement in 'patience' epochs, early stop
k_folds = 4

# model parameters
# the input dim of the model, you should not change the value
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')
_exp_name = "sample"

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
# Transforms: Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomResizedCrop(256),
    transforms.ColorJitter(brightness=0.12, saturation=0.12, contrast=0.12),
    transforms.RandomAffine(degrees=18, translate=(0.15, 0.15), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.autoaugment.AutoAugment(),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            ###############################################self.test = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            label = int(fname.split("/")[-1].split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label
"""
class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.out_ch = int(out_ch / 2)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_ch, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.Dropout(0.2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.net1(x)
        return  torch.cat([x,x1], dim=1)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            ResConv(64, 128),    # [128, 64, 64]
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]
            ResConv(128, 256),   # [256, 32, 32]
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]
            ResConv(256, 512),   # [512, 16, 16]
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]


            nn.Flatten(),
            nn.Linear(512*8*8, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11),
            
        )
    def forward(self, x):
        return nn.functional.softmax(self.net(x), dim = 0)
"""

##### Prepare dataset and model ##############################################
same_seeds(seed)
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
kfold = KFold(n_splits=k_folds, shuffle=True)
train_part = FoodDataset(os.path.join(dataset_path, "training"), tfm=train_tfm)

valid_part = FoodDataset(os.path.join(dataset_path, "validation"), tfm=test_tfm)

dataset = ConcatDataset([train_part, valid_part])

# create model, define a loss function, and optimizer
VGG11_train = torchvision.models.vgg11(weights = None).to(device)
optimizer_VGG11 = torch.optim.AdamW(VGG11_train.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler_VGG11 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_VGG11, T_max = 5, eta_min=1e-6)

ResNET_train = torchvision.models.resnet152(weights = None).to(device)
optimizer_ResNET = torch.optim.AdamW(ResNET_train.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler_ResNET = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ResNET, T_max = 5, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()
##### Training ###############################################################
stale = 0
best_acc = 0
start_time = time()
"""
for _ in range(5):
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=0, pin_memory=True, sampler=train_subsampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=0, pin_memory=True, sampler=valid_subsampler)
        for epoch in range(num_epochs):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            VGG11_train.train()

            # These are used to record information in training.
            train_loss = []
            train_accs = []

            for imgs, labels in tqdm(train_loader, position=0, leave=False):
                #writer.add_images("Img read", imgs)
                imgs, labels = imgs.to(device), labels.to(device)
                #imgs = imgs.half()
                #print(imgs.shape,labels.shape)

                # Forward the data. (Make sure data and model are on the same device.)
                logits = VGG11_train(imgs)

                # Calculate the cross-entropy loss.
                # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                loss = criterion(logits, labels)

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer_VGG11.zero_grad()

                # Compute the gradients for parameters.
                loss.backward()

                # Clip the gradient norms for stable training.
                grad_norm = nn.utils.clip_grad_norm_(VGG11_train.parameters(), max_norm=10)

                # Update the parameters with computed gradients.
                optimizer_VGG11.step()

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                train_loss.append(loss.item())
                train_accs.append(acc)
            scheduler_VGG11.step()

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            # Print the information.
            print(
                f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, lr = {scheduler_VGG11.get_last_lr()}, time = {(time() - start_time):5.2f}")
            writer.add_scalar("Train loss:", train_loss, epoch)
            writer.add_scalar("Train Acc:", train_acc, epoch)

            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            VGG11_train.eval()

            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader, position=0, leave=False):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = VGG11_train(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                # break

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # Print the information.
            #print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # update logs
            if valid_acc > best_acc:
                with open(f"./{_exp_name}_log.txt", "a"):
                    print(
                        f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            else:
                with open(f"./{_exp_name}_log.txt", "a"):
                    print(
                        f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            writer.add_scalar("Valid loss:", valid_loss, epoch)
            writer.add_scalar("Valid Acc:", valid_acc, epoch)

            # save models
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                # only save best to prevent output memory exceed error
                torch.save(VGG11_train.state_dict(), VGG11_path)
                best_acc = valid_acc
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    print(
                        f"No improvment {patience} consecutive epochs, early stopping")
                    break
            if stale > patience:
                break
        if stale > patience:
            break
    if stale > patience:
        stale = 0
        break

for _ in range(5):
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=0, pin_memory=True, sampler=train_subsampler)
        valid_loader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=0, pin_memory=True, sampler=valid_subsampler)
        for epoch in range(num_epochs):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            ResNET_train.train()

            # These are used to record information in training.
            train_loss = []
            train_accs = []

            for imgs, labels in tqdm(train_loader, position=0, leave=False):
                #writer.add_images("Img read", imgs)
                imgs, labels = imgs.to(device), labels.to(device)
                #imgs = imgs.half()
                #print(imgs.shape,labels.shape)

                # Forward the data. (Make sure data and model are on the same device.)
                logits = ResNET_train(imgs)

                # Calculate the cross-entropy loss.
                # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                loss = criterion(logits, labels)

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer_ResNET.zero_grad()

                # Compute the gradients for parameters.
                loss.backward()

                # Clip the gradient norms for stable training.
                grad_norm = nn.utils.clip_grad_norm_(ResNET_train.parameters(), max_norm=10)

                # Update the parameters with computed gradients.
                optimizer_ResNET.step()

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                train_loss.append(loss.item())
                train_accs.append(acc)
            scheduler_ResNET.step()

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            # Print the information.
            print(
                f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, lr = {scheduler_ResNET.get_last_lr()}, time = {(time() - start_time):5.2f}")
            writer.add_scalar("Train loss:", train_loss, epoch)
            writer.add_scalar("Train Acc:", train_acc, epoch)

            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            ResNET_train.eval()

            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader, position=0, leave=False):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = ResNET_train(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                # break

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # Print the information.
            #print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            # update logs
            if valid_acc > best_acc:
                with open(f"./{_exp_name}_log.txt", "a"):
                    print(
                        f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            else:
                with open(f"./{_exp_name}_log.txt", "a"):
                    print(
                        f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            writer.add_scalar("Valid loss:", valid_loss, epoch)
            writer.add_scalar("Valid Acc:", valid_acc, epoch)

            # save models
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch}, saving model")
                # only save best to prevent output memory exceed error
                torch.save(ResNET_train.state_dict(), ResNET_path)
                best_acc = valid_acc
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    print(
                        f"No improvment {patience} consecutive epochs, early stopping")
                    break
            if stale > patience:
                break
        if stale > patience:
            break
    if stale > patience:
        stale = 0
        break
"""

##### Testing ################################################################
# load data
test_set_test_tfm = FoodDataset(os.path.join(dataset_path, "test"), tfm=test_tfm)
test_set_train_tfm = FoodDataset(os.path.join(dataset_path, "test"), tfm=train_tfm)
test_loader_test_tfm = DataLoader(test_set_test_tfm, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)
test_loader_train_tfm = DataLoader(test_set_train_tfm, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)

# load model
#model_best = torchvision.models.vgg11_bn(weights=None).to(device)
VGG11_best = torchvision.models.vgg11(weights = None).to(device)
ResNET_best = torchvision.models.resnet152(weights = None).to(device)
VGG11_best.load_state_dict(torch.load(VGG11_path))
ResNET_best.load_state_dict(torch.load(ResNET_path))

# Make prediction.
VGG11_best.eval()
ResNET_best.eval()
prediction = []
test_preds = np.array([[]],)
test_preds_train = np.array([[]],dtype=float)
with torch.no_grad():
        ##Warning: close your eyes and don't look at my shit code :(
    for ((data0, _),(data1, _),(data2, _),(data3, _),(data4, _)) in tqdm(zip(test_loader_test_tfm, test_loader_train_tfm,test_loader_train_tfm,test_loader_train_tfm,test_loader_train_tfm), leave=True):
        pred0_V, pred0_R = VGG11_best(data0.to(device)).cpu(), ResNET_best(data1.to(device)).cpu()
        pred1_V, pred1_R = VGG11_best(data1.to(device)).cpu(), ResNET_best(data1.to(device)).cpu()
        pred2_V, pred2_R = VGG11_best(data2.to(device)).cpu(), ResNET_best(data2.to(device)).cpu()
        pred3_V, pred3_R = VGG11_best(data3.to(device)).cpu(), ResNET_best(data3.to(device)).cpu()
        pred4_V, pred4_R = VGG11_best(data4.to(device)).cpu(), ResNET_best(data4.to(device)).cpu()

        test_pred_V = np.sum((pred0_V*0.6, pred1_V*0.1, pred2_Vg*0.1, pred3_V*0.1, pred4_V*0.1), axis=0)
        test_pred_R = np.sum((pred0_R*0.6, pred1_R*0.1, pred2_R*0.1, pred3_R*0.1, pred4_R*0.1), axis=0)
        test_pred = np.sum((test_pred_V, test_pred_R), axis=0)
        test_label = np.argmax(test_pred, axis=1)
        prediction += test_label.squeeze().tolist()

# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_set_test_tfm)+1)]
df["Category"] = prediction
df.to_csv("./HW3/submission.csv", index=False)

#os.system("shutdown /s /t 60")
print(f"time = {(time() - start_time):5.2f}")