"""
Grokking Research Implementation
-------------------------------
This code implements a research framework for studying the grokking phenomenon in neural networks.
It includes comprehensive logging, visualization, and analysis tools.

Main components:
1. Data Generation and Processing
2. Model Architecture
3. Training Pipeline
4. Analysis and Visualization Tools
5. Experimental Configurations
"""

import torch
import einops
import tqdm.auto as tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd


class GrokkingExperiment:
    def __init__(self, config=None):
        """
        Initialize the grokking experiment with configuration parameters.

        Args:
            config (dict): Configuration parameters for the experiment
        """
        # Default configuration
        self.default_config = {
            'p': 113,  # Modulo base
            'frac_train': 0.3,  # Fraction of data for training
            'lr': 1e-3,  # Learning rate
            'wd': 1.0,  # Weight decay
            'betas': (0.9, 0.98),  # Adam optimizer betas
            'num_epochs': 25000,  # Number of training epochs
            'checkpoint_every': 1000,  # Checkpoint frequency
            'data_seed': 598,  # Random seed for data generation
            'model_seed': 999,  # Random seed for model initialization
            'device': 'cpu',  # Device to run on
            # Model configuration
            'model_config': {
                'n_layers': 1,
                'n_heads': 4,
                'd_model': 128,
                'd_head': 32,
                'd_mlp': 512,
                'act_fn': "relu",
                'normalization_type': None
            }
        }

        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        # Initialize storage for metrics
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.checkpoints = []
        self.checkpoint_epochs = []

        # Set random seeds
        torch.manual_seed(self.config['data_seed'])
        np.random.seed(self.config['data_seed'])

        # Setup experiment directory
        self.setup_experiment_dir()

    def setup_experiment_dir(self):
        """Create experiment directory with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f'experiments/grokking_{timestamp}')
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def generate_data(self):
        """Generate the modular arithmetic dataset"""
        p = self.config['p']

        # Create input vectors
        a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        equals_vector = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)

        # Create dataset and labels
        self.dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)
        self.labels = (self.dataset[:, 0] + self.dataset[:, 1]) % p

        # Split into train and test
        indices = torch.randperm(p * p)
        cutoff = int(p * p * self.config['frac_train'])
        self.train_indices = indices[:cutoff]
        self.test_indices = indices[cutoff:]

        self.train_data = self.dataset[self.train_indices]
        self.train_labels = self.labels[self.train_indices]
        self.test_data = self.dataset[self.test_indices]
        self.test_labels = self.labels[self.test_indices]

    def create_model(self):
        """Create and initialize the transformer model"""
        p = self.config['p']
        cfg = HookedTransformerConfig(
            d_vocab=p + 1,
            d_vocab_out=p,
            n_ctx=3,
            seed=self.config['model_seed'],
            device=self.config['device'],
            **self.config['model_config']
        )

        self.model = HookedTransformer(cfg)

        # Freeze bias parameters
        for name, param in self.model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        self.model.to(self.config['device'])

    def setup_training(self):
        """Setup optimizer and loss function"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['wd'],
            betas=self.config['betas']
        )

    def loss_fn(self, logits, labels):
        """Calculate loss with proper numerical stability"""
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()

    def calculate_accuracy(self, logits, labels):
        """Calculate prediction accuracy"""
        if len(logits.shape) == 3:
            logits = logits[:, -1]
        predictions = torch.argmax(logits, dim=-1)
        return (predictions == labels).float().mean().item()

    def train_epoch(self):
        """Train for one epoch and return metrics"""
        self.model.train()
        train_logits = self.model(self.train_data)
        train_loss = self.loss_fn(train_logits, self.train_labels)
        train_acc = self.calculate_accuracy(train_logits, self.train_labels)

        train_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return train_loss.item(), train_acc

    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        with torch.inference_mode():
            test_logits = self.model(self.test_data)
            test_loss = self.loss_fn(test_logits, self.test_labels)
            test_acc = self.calculate_accuracy(test_logits, self.test_labels)
        return test_loss.item(), test_acc

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for epoch in tqdm.tqdm(range(self.config['num_epochs'])):
            # Training step
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            # Store metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            # Checkpointing
            if ((epoch + 1) % self.config['checkpoint_every']) == 0:
                self.checkpoint_epochs.append(epoch)
                self.checkpoints.append(copy.deepcopy(self.model.state_dict()))
                self.save_checkpoint(epoch)
                self.create_visualizations(epoch)
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    def save_checkpoint(self, epoch):
        """Save model checkpoint and metrics"""
        checkpoint_path = self.exp_dir / f'checkpoint_{epoch + 1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
        }, checkpoint_path)

    def create_visualizations(self, epoch):
        """Create and save visualizations"""
        self.plot_learning_curves()
        self.plot_accuracy_curves()
        if epoch > 0:
            self.plot_loss_distributions()

    def plot_learning_curves(self):
        """Plot training and test loss curves"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(self.test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.exp_dir / 'learning_curves.png')
        plt.close()

    def plot_accuracy_curves(self):
        """Plot training and test accuracy curves"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy', color='green', alpha=0.7)
        plt.plot(self.test_accuracies, label='Test Accuracy', color='orange', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.exp_dir / 'accuracy_curves.png')
        plt.close()

    def plot_loss_distributions(self):
        """Plot loss distribution changes over time"""
        recent_train_losses = self.train_losses[-1000:]
        recent_test_losses = self.test_losses[-1000:]

        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=recent_train_losses, label='Training Loss', color='blue', alpha=0.7)
        sns.kdeplot(data=recent_test_losses, label='Test Loss', color='red', alpha=0.7)
        plt.xlabel('Loss')
        plt.ylabel('Density')
        plt.title('Loss Distributions (Last 1000 Epochs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.exp_dir / 'loss_distributions.png')
        plt.close()


def run_experiment(config=None):
    """Main function to run the experiment"""
    # Initialize experiment
    experiment = GrokkingExperiment(config)

    # Setup
    experiment.generate_data()
    experiment.create_model()
    experiment.setup_training()

    # Train
    experiment.train()

    return experiment


if __name__ == "__main__":
    # Example configuration
    config = {
        'num_epochs': 25000,
        'checkpoint_every': 1000,
        'lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Run experiment
    experiment = run_experiment(config)