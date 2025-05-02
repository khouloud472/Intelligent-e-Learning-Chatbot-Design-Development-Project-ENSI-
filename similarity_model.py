import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from typing import Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class EnhancedSimilarityPredictor(nn.Module):
    def __init__(self, input_dim: int = 1536):
        super().__init__()
        self.input_dim = input_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Input tensor must be of shape (batch_size, {self.input_dim}), got {x.shape}")
        attention_weights = self.attention(x)
        x = x * attention_weights
        return self.fc(x)

def generate_synthetic_data(embeddings: np.ndarray, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Génère des données synthétiques pour l'entraînement"""
    # Création de données réalistes à partir des embeddings
    true_samples = np.vstack([
        embeddings[np.random.choice(len(embeddings))] + np.random.normal(0, 0.2, embeddings.shape[1])
        for _ in range(n_samples // 2)
    ])
    
    noise_samples = np.random.normal(0, 1, (n_samples // 2, embeddings.shape[1]))
    
    X = np.vstack([true_samples, noise_samples])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    return X, y

def train_enhanced_model(
    embeddings: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> Tuple[nn.Module, Dict[str, list]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Entraînement sur {device}...")

    X, y = generate_synthetic_data(embeddings)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    model = EnhancedSimilarityPredictor(input_dim=embeddings.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'loss': loss.item()})

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        train_loss = train_loss / len(train_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            history[metric].append(val_metrics[metric])

        print(f"\nEpoch {epoch+1} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"ROC AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step(val_metrics['loss'])

    visualize_training_results(history, model, val_loader, device)
    return model, history

def evaluate_model(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    criterion: nn.Module
) -> Dict[str, float]:
    model.eval()
    loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels.unsqueeze(1)).item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)

    return {
        'loss': loss / len(data_loader.dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, all_preds)
    }

def visualize_training_results(
    history: Dict[str, list],
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device
):
    plt.figure(figsize=(20, 15))

    # Courbes de loss
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Évolution de la Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Métriques
    plt.subplot(2, 3, 2)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(history[metric], label=metric.capitalize())
    plt.title('Métriques de Classification')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Matrice de confusion
    plt.subplot(2, 3, 3)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')

    # Courbe ROC
    plt.subplot(2, 3, 4)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Courbe ROC')
    plt.xlabel('Faux positifs')
    plt.ylabel('Vrais positifs')
    plt.legend()
    plt.grid(True)

    # Courbe PR
    plt.subplot(2, 3, 5)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
    plt.title('Courbe Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    # Histogramme
    plt.subplot(2, 3, 6)
    sns.histplot([x for x, y in zip(all_preds, all_labels) if y == 1], color='blue', label='Positifs', kde=True, bins=20)
    sns.histplot([x for x, y in zip(all_preds, all_labels) if y == 0], color='red', label='Négatifs', kde=True, bins=20)
    plt.title('Distribution des Prédictions')
    plt.xlabel('Valeur prédite')
    plt.ylabel('Densité')
    plt.legend()

    plt.tight_layout()
    plt.show()
