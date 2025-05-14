import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import time
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CNN1D(nn.Module):
    """1D CNN for text classification based on BERT embeddings.
    
    This class can be initialized directly from a pandas DataFrame containing 
    embeddings and their corresponding labels.
    """
    MODEL_PATH = "../models/"
    
    def __init__(self, embedding_input, num_classes=None, label_column='user', 
                 embedding_prefix=None, dropout_rate=0.7, output_name=None, retrain=False):
        """
        Initialize the CNN model.
        
        Args:
            embedding_input: Can be either:
                - An integer representing the embedding dimension
                - A pandas DataFrame containing embeddings and labels
            num_classes: Number of classes for classification. If embedding_input is a DataFrame,
                         this will be inferred automatically.
            label_column: Name of the column containing the labels in the DataFrame. Required
                          if embedding_input is a DataFrame.
            embedding_prefix: Prefix of columns containing embeddings in the DataFrame.
                             If None, all columns except label_column are considered embeddings.
            dropout_rate: Dropout rate for regularization.
        """
        super(CNN1D, self).__init__()

        self.output_name = self.check_output_model_name(output_name)
        self.check_duplicate_model_name(self.output_name, retrain)
        
        if(torch.cuda.is_available()):
            print("[INFO]: Using GPU")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # Process input and determine embedding dimension and number of classes
        if isinstance(embedding_input, pd.DataFrame):
            if label_column is None:
                raise ValueError("label_column must be specified when initializing with DataFrame")
            
            self.df = embedding_input
            self.label_column = label_column
            
            # Determine embedding columns
            if embedding_prefix is not None:
                self.embedding_columns = [col for col in self.df.columns if col.startswith(embedding_prefix)]
            else:
                self.embedding_columns = [col for col in self.df.columns if col != label_column]
            
            # Determine embedding dimension
            embedding_dim = len(self.embedding_columns)
            
            # Determine number of classes
            if num_classes is None:
                num_classes = len(self.df[label_column].unique())
                
            # Store class names for later use
            self.class_names = sorted(self.df[label_column].unique())
            
            print(f"Detected {embedding_dim} embedding dimensions")
            print(f"Detected {num_classes} classes: {self.class_names}")
            
        else:
            # Direct initialization with dimension
            embedding_dim = embedding_input
            if num_classes is None:
                raise ValueError("num_classes must be specified when initializing with dimension")
            self.class_names = None
        
        # Store parameters
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Main CNN architecture

        # Multi-scale CNN architecture (better for capturing stylistic patterns at different scales)
        # Single word features
        conv1_output_channels = 128

        self.conv1_tiny = nn.Conv1d(1, conv1_output_channels, kernel_size=1)
        self.bn1_tiny = nn.BatchNorm1d(conv1_output_channels)

        # Small-scale features (3-gram)
        self.conv1_small = nn.Conv1d(1, conv1_output_channels, kernel_size=3, padding=1)
        self.bn1_small = nn.BatchNorm1d(conv1_output_channels)
        
        # Medium-scale features (5-gram)
        self.conv1_medium = nn.Conv1d(1, conv1_output_channels, kernel_size=5, padding=2)
        self.bn1_medium = nn.BatchNorm1d(conv1_output_channels)
        
        # Large-scale features (7-gram)
        self.conv1_large = nn.Conv1d(1, conv1_output_channels, kernel_size=7, padding=3)
        self.bn1_large = nn.BatchNorm1d(conv1_output_channels)
        
        # Combined features processing
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(conv1_output_channels*4, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after multiple pooling operations
        # pooled_size =  max(1, embedding_dim // 8)  # Ensure at least 1 to prevent zero-sized tensors
        
        # Global average pooling to handle variable length inputs better
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Attention mechanism for stylometry (helps focus on distinctive style markers)
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        # self.fc1 = nn.Linear(256, 512)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.fc2 = nn.Linear(512, num_classes)

        # NEW CLASSIFICATION HEAD
        self.classifier = nn.Sequential(
            # First layer
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            # Second layer
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.8),  # Gradually decrease dropout
            
            # Third layer
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.6),  # Further decrease dropout
            
            # Fourth layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.4),  # Even less dropout
            
            # Output layer
            nn.Linear(128, num_classes)
        )
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Store best model state
        self.best_model_state = None
        self.best_val_acc = 0.0
        
    def forward(self, x):
        # Input shape: [batch_size, embedding_dim]
        # Reshape for 1D convolution [batch_size, channels, embedding_dim]
        x = x.unsqueeze(1)
        
        # Multi-scale feature extraction
        x_tiny = self.relu(self.bn1_tiny(self.conv1_tiny(x)))
        x_small = self.relu(self.bn1_small(self.conv1_small(x)))
        x_medium = self.relu(self.bn1_medium(self.conv1_medium(x)))
        x_large = self.relu(self.bn1_large(self.conv1_large(x)))
        
        # Combine features from different scales (use the one that matches your architecture)
        x = torch.cat([x_tiny, x_small, x_medium, x_large], dim=1)  # Concatenation along channel dimension
        
        # Apply pooling to the combined features
        x = self.pool1(x)
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Global average pooling 
        x_pooled = self.global_avg_pool(x).squeeze(-1)
        
        # Apply attention 
        attention_weights = self.attention(x_pooled)
        x_attended = x_pooled * attention_weights
        
        # Flatten for fully connected layer
        x_attended = x_attended.view(x.size(0), -1)
        
        # Classification head
        # x_attended = self.relu(self.fc1(x_attended))
        # x_attended = self.dropout(x_attended)
        # x_attended = self.fc2(x_attended)

        output = self.classifier(x_attended)
        
        return output
        
    def prepare_data(self, test_size=0.2, val_size=0.25, batch_size=32, random_state=42):
        """
        Prepare train, validation, and test data loaders from the DataFrame.
        
        Args:
            test_size: Fraction of data to use for testing
            val_size: Fraction of remaining data to use for validation
            batch_size: Batch size for data loaders
            random_state: Random seed for reproducibility
            
        Returns:
            train_loader, val_loader, test_loader
        """
        if not hasattr(self, 'df'):
            raise ValueError("Model was not initialized with a DataFrame")
        
        # Extract embeddings and labels
        X = self.df[self.embedding_columns].values
        y = self.df[self.label_column].values
        
        # If labels are not numeric, encode them
        if not np.issubdtype(y.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.label_encoder = label_encoder
            print(f"Encoded labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Check class distribution
        train_class_counts = np.bincount(y_train)
        val_class_counts = np.bincount(y_val)
        test_class_counts = np.bincount(y_test)
        
        print("Class distribution in training set:", train_class_counts)
        print("Class distribution in validation set:", val_class_counts)
        print("Class distribution in test set:", test_class_counts)
            
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDatasets
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Calculate class weights
        class_weights = 1.0 / torch.tensor(train_class_counts, dtype=torch.float)
        sample_weights = class_weights[y_train_tensor]

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
        
    def train_model(self, train_loader, val_loader, criterion=None, optimizer=None, 
                num_epochs=10, learning_rate=0.001):
        """Train the CNN model."""
        
        # Set up the device
        self.to(self.device)
        
        # Set up default criterion and optimizer if not provided
        if criterion is None:
            # Calculate class weights if needed
            class_counts = torch.bincount(torch.tensor([y for _, y in train_loader.dataset]))
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Add learning rate scheduler
        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
        
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-6)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
            for embeddings, labels in train_bar:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(embeddings)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * embeddings.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)
            
            # Validation phase
            self.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
                for embeddings, labels in val_bar:
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self(embeddings)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item() * embeddings.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
            epoch_val_loss = running_loss / len(val_loader.dataset)
            epoch_val_acc = correct / total
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
            print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
            
            # Save the best model
            if epoch_val_acc > self.best_val_acc:
                self.best_val_acc = epoch_val_acc
                self.best_model_state = self.state_dict().copy()
                print(f'  New best model saved with validation accuracy: {self.best_val_acc:.4f}')

            # after validation:
            #scheduler.step(epoch_val_acc)
            scheduler.step()
        
        # Load the best model
        self.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
    def evaluate(self, test_loader, class_names=None):
        """Evaluate the trained model."""
        
        # Set up the device
        self.to(self.device)
        
        # Use class names from initialization if not provided
        if class_names is None and hasattr(self, 'class_names'):
            class_names = self.class_names
        
        self.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(test_loader, desc='Evaluating'):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self(embeddings)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, 
                                    target_names=class_names if class_names else None, 
                                    output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return report_df, cm, all_preds, all_labels
    
    def train_and_evaluate(self, test_size=0.2, val_size=0.2, batch_size=32, 
                     num_epochs=10, learning_rate=0.001, 
                     criterion=None, optimizer=None, random_state=42, 
                     plot_results=True):
        """
        Complete pipeline that runs data preparation, model training, and evaluation in sequence.
        
        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            val_size: Fraction of remaining data to use for validation (default: 0.2)
            batch_size: Batch size for data loaders (default: 32)
            num_epochs: Number of training epochs (default: 10)
            learning_rate: Learning rate for optimizer if not provided (default: 0.001)
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            random_state: Random seed for reproducibility (default: 42)
            output_model_name: Path to save the trained model (default: None - don't save)
            plot_results: Whether to plot training history and confusion matrix (default: True)
            
        Returns:
            Dictionary containing:
            - 'model': Trained model
            - 'history': Training history
            - 'report': Classification report
            - 'confusion_matrix': Confusion matrix
            - 'predictions': Model predictions on test set
            - 'true_labels': True labels of test set
            - 'data_loaders': (train_loader, val_loader, test_loader)
        """
        
        # Step 1: Prepare data
        print("\n--- Preparing Data ---")
        train_loader, val_loader, test_loader = self.prepare_data(
            test_size=test_size,
            val_size=val_size,
            batch_size=batch_size,
            random_state=random_state
        )
        
        # Step 2: Train model
        print("\n--- Training Model ---")
        history = self.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Step 3: Evaluate model
        print("\n--- Evaluating Model ---")
        report, cm, predictions, true_labels = self.evaluate(
            test_loader=test_loader,
        )
        
        # Print evaluation results
        print("\n--- Evaluation Results ---")
        print(report)
        
        # Save the model
        self.save_model()
        
        # Optional: Plot results
        if plot_results:
            self.plot_training_results(history, cm)
        
        # Return all results
        return {
            'model': self,
            'history': history,
            'report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'data_loaders': (train_loader, val_loader, test_loader)
        }

    def _get_logits(self, X):
        """
        Shared routine: preprocess X, 
        run it through the model, and return raw logits.
        """
        # Set up the device
        self.to(self.device)
        
        # Convert input to appropriate format
        if isinstance(X, pd.DataFrame):
            if hasattr(self, 'embedding_columns'):
                X = X[self.embedding_columns].values
            else:
                # If embedding_columns not defined, use all columns
                X = X.values
        
        # Convert to tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Make sure we have a batch dimension
        if X.dim() == 1:
            X = X.unsqueeze(0)
            
        # Move to device
        X = X.to(self.device)
        
        # Make predictions
        self.eval()
        with torch.no_grad():
            logits = self(X)
        
        return logits
    
    def predict(self, X):
        """
        Return hard class predictions (indices).
        """
        logits = self._get_logits(X)
        _, preds = torch.max(logits, dim=1)
        return preds.cpu().numpy()
    
    def predict_proba(self, X):
        """
        Return class‐membership probabilities.
        """
        logits = self._get_logits(X)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    def plot_training_results(self, history, confusion_matrix=None):
        """
        Plot training history and confusion matrix.
        
        Args:
            history: Dictionary containing training history with keys:
                    'train_losses', 'val_losses', 'train_accs', 'val_accs'
            confusion_matrix: Confusion matrix to plot (optional)
        """
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accs'], label='Train Accuracy')
        plt.plot(history['val_accs'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrix if provided
        if confusion_matrix is not None:
            plt.figure(figsize=(10, 8))
            class_names = self.class_names if hasattr(self, 'class_names') else None
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
    
    def save_model(self):
        """Save the model to a file."""

        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)

        output_path = os.path.join(self.MODEL_PATH, self.output_name)

        torch.save({
            'model_state_dict': self.state_dict(),
            'class_names': self.class_names if hasattr(self, 'class_names') else None
        }, output_path)

    
    def check_output_model_name(self, name):
        """Check if the model name is valid."""
        if name:
            return self.check_prefix_extension(name, "embed_", ".pth")
        else:
            return "embed_" + str(calendar.timegm(time.gmtime())) + ".pth"
        
    def check_prefix_extension(self, name, prefix, extension):
        """Check if the name has the correct prefix and extension."""
        if not name.startswith(prefix):
            name = prefix + name
        if not name.endswith(extension):
            name = name + extension
        return name
    
    def check_duplicate_model_name(self, name, retrain):
        """Check if the model name already exists."""

        if os.path.exists(os.path.join(self.MODEL_PATH, name)) and not retrain:
            raise ValueError(f"Model with name {name} already exists. Allow retraining to overrite.")
    