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
import time
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd
from contextlib import nullcontext


class GrokkingExperiment:
    def __init__(self, config=None):
        """Initialize the experiment"""
        self.default_config = {
            'p': 113,  # Modulo base
            'frac_train': 0.3,  # Training data fraction
            'lr': 1e-3,  # Learning rate
            'wd': 1.0,  # Weight decay
            'betas': (0.9, 0.98),  # Adam optimizer betas
            'num_epochs': 25000,  # Training epochs
            'checkpoint_every': 5000,  # Checkpoint frequency
            'weight_track_every': 5000,  # Weight tracking frequency
            'batch_size': 2048,  # Batch size
            'data_seed': 598,  # Data generation seed
            'model_seed': 999,  # Model initialization seed
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            # Model configuration
            'model_config': {
                'n_layers': 1,
                'n_heads': 4,
                'd_model': 64,  # Reduced model dimension
                'd_head': 16,  # Reduced head dimension
                'd_mlp': 256,  # Reduced MLP dimension
                'act_fn': "relu",
                'normalization_type': None
            }
        }

        # Update config with any overrides
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        # Initialize metrics storage
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        # Initialize weight tracking storage
        self.weight_tracking = {
            'epochs': [],
            'embedding': [],  # Token embedding weights
            'positional': [],  # Positional embedding weights
            'query': [],  # Query weights
            'key': [],
            'value': [],  # Value weights
            'output': [],  # Output projection weights
            'mlp_in': [],  # MLP input weights
            'mlp_out': [],  # MLP output weights
            'unembedding': []  # Unembedding weights
        }

        # Set random seeds
        self._set_random_seeds()

        # Setup directories
        self.setup_directories()

    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config['model_seed'])
        np.random.seed(self.config['data_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['model_seed'])

    def setup_directories(self):
        """Setup experiment directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f'experiments/grokking_{timestamp}')

        # Create subdirectories
        self.dirs = {
            'weights': self.exp_dir / 'weights',
            'heatmaps': self.exp_dir / 'heatmaps',
            'metrics': self.exp_dir / 'metrics',
            'checkpoints': self.exp_dir / 'checkpoints'
        }

        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def generate_data(self):
        """Generate modular arithmetic dataset"""
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

        # Move to device
        self.train_data = self.dataset[self.train_indices].to(self.config['device'])
        self.train_labels = self.labels[self.train_indices].to(self.config['device']).long()
        self.test_data = self.dataset[self.test_indices].to(self.config['device'])
        self.test_labels = self.labels[self.test_indices].to(self.config['device']).long()

    def create_model(self):
        """Create and initialize the transformer model"""
        p = self.config['p']

        # Create model configuration
        cfg = HookedTransformerConfig(
            d_vocab=p + 1,
            d_vocab_out=p,
            n_ctx=3,
            seed=self.config['model_seed'],
            device=self.config['device'],
            **self.config['model_config']
        )

        # Initialize model
        self.model = HookedTransformer(cfg)

        # Freeze bias parameters
        for name, param in self.model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        self.model.to(self.config['device'])

    def setup_training(self):
        """Setup optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['wd'],
            betas=self.config['betas']
        )

    def loss_fn(self, logits, labels):
        """Calculate cross-entropy loss"""
        if len(logits.shape) == 3:
            logits = logits[:, -1]  # Take the last token's predictions
        return torch.nn.functional.cross_entropy(logits, labels)

    def train_epoch(self):
        """Optimized training for one epoch"""
        self.model.train()
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            train_logits = self.model(self.train_data)
            if len(train_logits.shape) == 3:
                train_logits = train_logits[:, -1]  # Take the last token's predictions
            train_loss = self.loss_fn(train_logits, self.train_labels)
            train_acc = self.calculate_accuracy(train_logits, self.train_labels)

        train_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return train_loss.item(), train_acc

    def evaluate(self):
        """Optimized evaluation"""
        self.model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            test_logits = self.model(self.test_data)
            if len(test_logits.shape) == 3:
                test_logits = test_logits[:, -1]  # Take the last token's predictions
            test_loss = self.loss_fn(test_logits, self.test_labels)
            test_acc = self.calculate_accuracy(test_logits, self.test_labels)
        return test_loss.item(), test_acc

    def calculate_accuracy(self, logits, labels):
        """Calculate accuracy"""
        predictions = torch.argmax(logits, dim=-1)
        return (predictions == labels).float().mean().item()

    @torch.no_grad()
    def track_weights(self, epoch):
        """More efficient weight tracking"""
        # Only track if it's a visualization epoch
        if epoch % (self.config['weight_track_every'] * 5) != 0:
            return

        save_dir = self.dirs['weights'] / f'epoch_{epoch}'
        save_dir.mkdir(exist_ok=True)

        # Extract and save weights more efficiently
        weights = {
            'embedding': self.model.embed.W_E,
            'positional': self.model.pos_embed.W_pos,
            'query': self.model.blocks[0].attn.W_Q,
            'key': self.model.blocks[0].attn.W_K,
            'value': self.model.blocks[0].attn.W_V,
            'output': self.model.blocks[0].attn.W_O,
            'mlp_in': self.model.blocks[0].mlp.W_in,
            'mlp_out': self.model.blocks[0].mlp.W_out,
            'unembedding': self.model.unembed.W_U
        }

        # Batch process weights
        for name, weight in weights.items():
            self.weight_tracking[name].append(weight.cpu().numpy())
            np.save(save_dir / f'{name}.npy', self.weight_tracking[name][-1])

        self.weight_tracking['epochs'].append(epoch)

    def train(self):
        """Main training loop"""
        print("Starting training...")
        pbar = tqdm.tqdm(range(self.config['num_epochs']), desc="Training")

        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            pbar.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Test Loss": f"{test_loss:.4f}",
                "Train Acc": f"{train_acc:.4f}",
                "Test Acc": f"{test_acc:.4f}"
            })

            if epoch % self.config['weight_track_every'] == 0:
                self.track_weights(epoch)

            if epoch % self.config['checkpoint_every'] == 0:
                checkpoint_path = self.dirs['checkpoints'] / f'checkpoint_epoch_{epoch}.pt'
                torch.save(self.model.state_dict(), checkpoint_path)

        print("Training complete!")

        # Create visualizations
    def create_grokking_visualizations(self):
        """Create comprehensive visualizations of the grokking phenomenon"""

        # Create visualization directory
        viz_dir = self.exp_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # 1. Training Dynamics Plot (Loss and Accuracy)
        plt.figure(figsize=(15, 6))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.semilogy(self.train_losses, label='Train Loss', alpha=0.8)
        plt.semilogy(self.test_losses, label='Test Loss', alpha=0.8)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.legend()
        plt.title('Training and Test Loss Over Time')

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', alpha=0.8)
        plt.plot(self.test_accuracies, label='Test Accuracy', alpha=0.8)
        plt.grid(True, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Test Accuracy Over Time')

        plt.tight_layout()
        plt.savefig(viz_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Grokking Phase Transition Analysis
        # Find the grokking point (where test accuracy significantly improves)
        test_acc_array = np.array(self.test_accuracies)
        diff = np.diff(test_acc_array)
        grokking_point = np.argmax(diff) + 1

        plt.figure(figsize=(12, 6))
        plt.plot(test_acc_array, label='Test Accuracy', alpha=0.8)
        plt.axvline(x=grokking_point, color='r', linestyle='--',
                    label=f'Grokking Point (epoch {grokking_point})')
        plt.grid(True, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title('Grokking Phase Transition')
        plt.legend()
        plt.savefig(viz_dir / 'grokking_transition.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Loss Landscape Evolution
        plt.figure(figsize=(15, 5))
        epochs = range(0, len(self.train_losses), len(self.train_losses) // 10)
        for i, epoch in enumerate(epochs):
            plt.subplot(1, 2, 1)
            plt.semilogy(epoch, self.train_losses[epoch], 'bo', alpha=0.5)
            plt.semilogy(epoch, self.test_losses[epoch], 'ro', alpha=0.5)

            plt.subplot(1, 2, 2)
            plt.plot(epoch, self.train_accuracies[epoch], 'bo', alpha=0.5)
            plt.plot(epoch, self.test_accuracies[epoch], 'ro', alpha=0.5)

        plt.subplot(1, 2, 1)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Evolution at Key Points')

        plt.subplot(1, 2, 2)
        plt.grid(True, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Evolution at Key Points')

        plt.tight_layout()
        plt.savefig(viz_dir / 'evolution_points.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Phase Space Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.train_losses, self.test_losses,
                    c=range(len(self.train_losses)), cmap='viridis',
                    alpha=0.5)
        plt.colorbar(label='Epoch')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Training Loss (log scale)')
        plt.ylabel('Test Loss (log scale)')
        plt.title('Loss Phase Space')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(viz_dir / 'phase_space.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save numerical analysis
        analysis = {
            'grokking_epoch': int(grokking_point),  # Convert to Python int
            'final_train_loss': float(self.train_losses[-1]),  # Convert to Python float
            'final_test_loss': float(self.test_losses[-1]),
            'final_train_acc': float(self.train_accuracies[-1]),
            'final_test_acc': float(self.test_accuracies[-1]),
            'total_epochs': int(len(self.train_losses))
        }

        with open(viz_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4)

if __name__ == "__main__":
    # Configuration
    config = {
        'num_epochs': 25000,
        'checkpoint_every': 5000,
        'weight_track_every': 5000,
        'batch_size': 2048,
        'lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_config': {
            'n_layers': 1,
            'n_heads': 2,
            'd_model': 32,
            'd_head': 16,
            'd_mlp': 128,
            'act_fn': "relu",
            'normalization_type': None
        }
    }

    # Run experiment
    experiment = GrokkingExperiment(config)

    # Enable GPU optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    experiment.generate_data()
    experiment.create_model()
    experiment.setup_training()
    experiment.train()

    # Create visualizations after training
    print("Creating visualizations...")
    experiment.create_grokking_visualizations()
    print("Visualizations saved!")

