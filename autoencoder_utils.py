import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoder(nn.Module):
    
    def __init__(self, latent_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, latent_dim)

        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(True),
            nn.Linear(50, 100),
            nn.ReLU(True),
            nn.Linear(100, 28),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode_numpy(self, X):
        """
        Encode the input data using the encoder.
        
        Args:
            X (numpy.ndarray): Input data to encode.
            
        Returns:
            numpy.ndarray: Encoded data.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.encoder(X_tensor).numpy()
        return encoded


def train_autoencoder(
    model, X_train, X_val, epochs, batch_size, learning_rate=1e-3, patience=5
):
    """
    Train the model using numpy data with early stopping.
    
    Args:
        model (torch.nn.Module): The model to train.
        X_train (numpy.ndarray): Training data as numpy array.
        X_val (numpy.ndarray): Validation data as numpy array.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for the DataLoader.
        patience (int): Number of epochs to wait for improvement in validation loss.
        
    Returns:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    # Early stopping variables
    best_val_loss = float('inf')  # Initialize best loss to infinity
    patience_counter = 0          # Track epochs without improvement

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best validation loss
            patience_counter = 0     # Reset patience counter
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1   # Increment patience counter
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model state
    model.load_state_dict(best_model_state)

    return train_losses, val_losses