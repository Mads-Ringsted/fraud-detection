import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import AutoEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

EPOCHS = 250
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def load_data():
    data = pd.read_csv('creditcard.csv')

    data = data.drop(['Time', 'Amount'], axis=1)
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    return X, y

X, y = load_data()

X_non_fraud = X[y == 0].iloc[:2000]#.sample(5000)
X_fraud = X[y == 1]


scaler = MinMaxScaler()
X = scaler.fit_transform(X_non_fraud)
#X = X_non_fraud.values

model = AutoEncoder()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

rows, cols = X.shape
rows = int(rows*0.8)
X_train = torch.tensor(X[:rows], dtype=torch.float32)
X_val = torch.tensor(X[rows:], dtype=torch.float32)

train_dataset = TensorDataset(X_train, X_train)
val_dataset = TensorDataset(X_val, X_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_losses = []
val_losses = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# save the model
torch.save(model.state_dict(), 'model.pth')

# Create a plot of the latent space representation of both the non-fraud and fraud data
model.eval()
X = torch.tensor(scaler.transform(X_fraud.iloc[:2000]), dtype=torch.float32)
with torch.no_grad():
    latent_space = model.encoder(X).numpy()
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c='r', label='Fraud')

X = torch.tensor(scaler.transform(X_non_fraud), dtype=torch.float32)
with torch.no_grad():
    latent_space = model.encoder(X).numpy()
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c='b', label='Non-Fraud')
plt.show()
