# Download timit from: https://drive.google.com/uc?id=1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import gc

# (Const) Parameters
VAL_RATIO = 0.2
BATCH_SIZE = 16
EPOCH = 80
LEARNING_RATE = 0.0001
SEED = 1
DATA_ROOT = './HW2/datasets/'
MODEL_PATH = './HW2/models/model.ckpt'
CONCAT_FRAMES = 21

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {DEVICE}')


# Load Data
print('Loading data ...')
train = np.load(DATA_ROOT + 'train_11.npy')
train_label = np.load(DATA_ROOT + 'train_label_11.npy')
test = np.load(DATA_ROOT + 'test_11.npy')
print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))


# Create Dataset
class TIMITDataset(Dataset):
    def __init__(self, X, y=None, concat_frames=21):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int32)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            self.test = torch.cat(
                (torch.zeros(self.data[idx].size()), self.data[idx]))
            print(self.data.size())
            print(self.label.size())
            print(self.test.size())
            print("after:")
            self.data = torch.nn.functional.pad(
                self.data, (0, 0, 1, 1), "constant", 0)
            print(self.data.size())
            return self.data[idx], self.label[idx]

        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data.
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], \
    train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))


# Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here.
train_set = TIMITDataset(train_x, train_y, CONCAT_FRAMES)
val_set = TIMITDataset(val_x, val_y, CONCAT_FRAMES)
# only shuffle the training data
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


# Cleanup the unneeded variables to save memory.
# notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**
del train, train_label, train_x, train_y, val_x, val_y
gc.collect()


# Create Model： Define model architecture, you are encouraged to change and experiment with the model architecture.
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(429, 2048),
            nn.Sigmoid(),
            # nn.Linear(1024,512),
            # nn.Sigmoid(),
            nn.Linear(2048, 256),
            nn.Sigmoid(),
            nn.Linear(256, 39)
        )

    def forward(self, x):
        return self.layers(x)


# Training
# fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(SEED)


# create model, define a loss function, and optimizer
model = Classifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, EPOCH, eta_min=0)

# start training
best_acc = 0.0
for epoch in range(EPOCH):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        # get the index of the class with the highest probability
        _, train_pred = torch.max(outputs, 1)
        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()
    scheduler.step()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                # get the index of the class with the highest probability
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, EPOCH, train_acc/len(train_set), train_loss/len(
                    train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print('saving model with acc {:.3f}'.format(
                    best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, EPOCH, train_acc /
            len(train_set), train_loss/len(train_loader)
        ))


# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), MODEL_PATH)
    print('saving model at last epoch')


# Testing: Create a testing dataset, and load model from the saved checkpoint.
# create testing dataset
test_set = TIMITDataset(test, None, CONCAT_FRAMES)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))


# Make prediction.
predict = []
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        # get the index of the class with the highest probability
        _, test_pred = torch.max(outputs, 1)

        for y in test_pred.cpu().numpy():
            predict.append(y)


# Write prediction to a CSV file.: After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
