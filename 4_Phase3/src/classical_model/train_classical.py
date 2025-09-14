import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import time
import pickle
import joblib
import matplotlib.pyplot as plt
import argparse
import os

class ClassicalLSTMModel(nn.Module):
    """LSTM model for multi-step time-series forecasting"""
    def __init__(self, n_features, n_lstm_units=128, num_layers=4, output_len=72):
        super(ClassicalLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.classifier = nn.Linear(n_lstm_units, output_len)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_lstm_output = lstm_out[:, -1, :]
        output = self.classifier(final_lstm_output)
        return torch.sigmoid(output)


def train_model_pytorch(model, train_loader, val_loader, epochs=25, patience=5, model_save_path=None):
    """Train the PyTorch model"""
    print("Starting training with PyTorch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    
    if model_save_path is None:
        model_save_path = 'models/classical.pth'

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_correct, train_samples = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_samples += y_batch.size(0) * y_batch.size(1)

        train_loss /= train_samples
        train_acc = train_correct / train_samples
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_samples += y_batch.size(0) * y_batch.size(1)

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Duration: {epoch_duration:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print("  -> Validation loss improved. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

    model.load_state_dict(torch.load(model_save_path))
    return model, history


def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (per timestep)')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('results/classical/classical_training_history_multistep.png', dpi=300, bbox_inches='tight')
    plt.show()


def load_training_data(data_dir):
    """Load preprocessed training data"""
    print(f"Loading training data from {data_dir}")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    loc_train = np.load(os.path.join(data_dir, 'loc_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    loc_val = np.load(os.path.join(data_dir, 'loc_val.npy'))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, y_train, loc_train, X_val, y_val, loc_val


def main():
    parser = argparse.ArgumentParser(description="Train classical LSTM model")
    parser.add_argument("--data_dir", required=True, help="Directory containing preprocessed data")
    parser.add_argument("--model_save_path", default="classical.pth", 
                       help="Path to save the best model")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--n_lstm_units", type=int, default=128, help="Number of LSTM units")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of LSTM layers")
    parser.add_argument("--n_features", type=int, default=9, help="Number of input features")
    parser.add_argument("--output_len", type=int, default=72, help="Output sequence length")
    parser.add_argument("--plot_history", action="store_true", help="Plot training history")
    
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, loc_train, X_val, y_val, loc_val = load_training_data(args.data_dir)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = ClassicalLSTMModel(
        n_features=args.n_features,
        n_lstm_units=args.n_lstm_units,
        num_layers=args.num_layers,
        output_len=args.output_len
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    model, history = train_model_pytorch(
        model, train_loader, val_loader, 
        epochs=args.epochs, patience=args.patience, 
        model_save_path=args.model_save_path
    )
    
    # Save training history
    history_path = args.model_save_path.replace('.pth', '_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(dict(history), f)
    print(f"Training history saved to {history_path}")
    
    # Plot training history if requested
    if args.plot_history:
        plot_path = args.model_save_path.replace('.pth', '_history.png')
        plot_training_history(history, plot_path)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
