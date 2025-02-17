import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data import fetch_data, prepare_data, split_data

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, axis=-1)
        self.x = x.astype(np.float32)
        self.y = np.array(y).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Model(nn.Module):
    def __init__(self, input_size = 1, output_size = 1, hidden_layer_size = 32, num_layers = 2, dropout = 0.2):
        super(Model, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_final = nn.Linear(num_layers*hidden_layer_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(x)
        x = self.relu(x)

        lstm_out, (hidden, cell) = self.lstm(x)
        x = hidden.permute(1, 0, 2).reshape(batch_size, -1)

        x = self.dropout(x)
        prediction = self.linear_final(x)
        return prediction[:,-1]

def run_epoch(model, dataloader, criterion, optimizer, scheduler, is_training = False):
    loss = 0
    if is_training:
        model.train()
    else:
        model.eval()

    for i, (x,y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()
        batch_size = x.shape[0]

        x = x.to('cpu')
        y = y.to('cpu')

        output = model(x)
        error = criterion(output, y)

        if is_training:
            error.backward()
            optimizer.step()

        loss += (error.detach().item()/batch_size)

    return loss, scheduler.get_last_lr()[0]

batch_size = 64
num_epochs = 100
window = 20
scheduler_step_size = 40

df = None
while df is None or df.empty:
    symbol = input('Enter stock symbol: ')
    df = fetch_data(f'{symbol}')

x,y = prepare_data(df,window)
x_train, x_val, y_train, y_val = split_data(x = x, y = y, scale = 0.8)

dataset_train = TimeSeriesDataset(x_train, y_train)
dataset_val = TimeSeriesDataset(x_val, y_val)
dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = True)

model = Model()
model = model.to('cpu')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size, gamma = 0.1)

for epoch in range(num_epochs):
    loss_train, lr_train = run_epoch(model, dataloader_train, criterion, optimizer, scheduler, is_training = True)
    loss_val, lr_val = run_epoch(model, dataloader_val, criterion, optimizer, scheduler)
    scheduler.step()

    #print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'.format(epoch + 1, num_epochs, loss_train, loss_val, lr_train))

dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = False)
model.eval()

predictions = np.array([])
for i, (x,y) in enumerate(dataloader_train):
    x = x.to('cpu')
    out = model(x)
    out = out.cpu().detach().numpy()
    predictions = np.concatenate((predictions, out))

for i, (x,y) in enumerate(dataloader_val):
    x = x.to('cpu')
    out = model(x)
    out = out.cpu().detach().numpy()
    predictions = np.concatenate((predictions, out))

predictions = (predictions * df['Close'].std().item()) + df['Close'].mean().item()

plt.figure(figsize=(20,12))
plt.plot(df.index, df['Close'], label = 'Closing Price', color = 'black')
plt.plot(df.index[window:], predictions, label = 'Predicted Closing Price', color = 'red')
plt.xlabel('Date')
plt.ylabel('Price ($USD)')
plt.legend()
plt.show()