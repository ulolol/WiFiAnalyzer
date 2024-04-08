#Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Load and preprocess data
data = pd.read_csv('your_data.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Hour'] = data['Timestamp'].dt.hour
data[['Latitude', 'Longitude']] = data['GPS Coordinates'].str.split(',', expand=True).astype(float)
data.drop(columns=['GPS Coordinates'], inplace=True)
data['Signal Strength'] = data['Signal Strength'].str.extract('(-\d+)').astype(int)

scaler = MinMaxScaler()
data['Signal Strength'] = scaler.fit_transform(data[['Signal Strength']])

features = data[['Hour', 'Latitude', 'Longitude']].values
labels = data[['Signal Strength']].values

train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

#Define PyTorch Lightning model
class WifiSignalStrengthModel(pl.LightningModule):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1):
        super(WifiSignalStrengthModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.loss_function = nn.MSELoss()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1])

    def training_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self(features)
        loss = self.loss_function(predictions, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

#Convert to PyTorch tensors and create DataLoaders
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)

#Initialize model and train
model = WifiSignalStrengthModel()
trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=20)
trainer.fit(model, train_dataloader)

#Save the model
torch.save(model.state_dict(), 'wifi_signal_strength_model.pth')

#Example prediction (replace with actual data)
test_data = torch.tensor([[10, 37.7749, -122.4194]], dtype=torch.float32)
model.eval()  #Set model to evaluation mode
prediction = model(test_data)
print(f"Predicted signal strength: {prediction.item()}")