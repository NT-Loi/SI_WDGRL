import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset 
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Feature extractor network."""
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """Domain critic network."""
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class WDGRL():
    def __init__(self, input_dim: int=2, generator_hidden_dims: List[int]=[32, 16, 8, 4, 2], critic_hidden_dims: List[int]=[32, 16, 8, 4, 2],
                 gamma: float = 0.1, _lr_generator: float = 1e-2, _lr_critic: float = 1e-2, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.gamma = gamma
        self.device = device
        self.generator = Generator(input_dim, generator_hidden_dims).to(self.device)
        self.critic = Critic(generator_hidden_dims[-1], critic_hidden_dims).to(self.device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=_lr_generator)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=_lr_critic)
    
    def compute_gradient_penalty(self, source_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty."""
        # if source_data.size(0) > target_data.size(0):
        #     ms = source_data.size(0)
        #     mt = target_data.size(0)
        #     gradient_penalty = 0
        #     for _ in range(0, ms, mt):
        #         source_chunk = source_data[_:_+mt]
        #         target_chunk = target_data
        #         alpha = torch.rand(target_chunk.size(0), 1).to(self.device)
        #         interpolates = (alpha * source_chunk + ((1 - alpha) * target_chunk)).requires_grad_(True)
                
        #         # Domain critic outputs
        #         dc_output = self.critic(interpolates)
                
        #         # Compute gradients
        #         gradients = torch.autograd.grad(
        #             outputs=dc_output,
        #             inputs=interpolates,
        #             grad_outputs=torch.ones_like(dc_output).to(self.device),
        #             create_graph=True,
        #             retain_graph=True,
        #             only_inputs=True,
        #         )
        #         gradients = gradients[0]
        #         gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        #     if ms % mt != 0:
        #         source_chunk = source_data[ms-mt:]
        #         perm = torch.randperm(mt)
        #         idx = perm[:ms % mt]
        #         target_chunk = target_data[idx]
        #         alpha = torch.rand(target_chunk.size(0), 1).to(self.device)
        #         interpolates = (alpha * source_chunk + ((1 - alpha) * target_chunk)).requires_grad_(True)
                
        #         # Domain critic outputs
        #         dc_output = self.critic(interpolates)
                
        #         # Compute gradients
        #         gradients = torch.autograd.grad(
        #             outputs=dc_output,
        #             inputs=interpolates,
        #             grad_outputs=torch.ones_like(dc_output).to(self.device),
        #             create_graph=True,
        #             retain_graph=True,
        #             only_inputs=True,
        #         )
        #         gradients = gradients[0]
        #         gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        #     return gradient_penalty / ((ms // mt) + (ms % mt != 0)) 
        
        # For balanced batch
        alpha = torch.rand(source_data.size(0), 1).to(self.device)
        interpolates = (alpha * source_data + ((1 - alpha) * target_data)).requires_grad_(True)
        
        # Domain critic outputs
        dc_output = self.critic(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=dc_output,
            inputs=interpolates,
            grad_outputs=torch.ones_like(dc_output).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def train(self, source_loader: DataLoader, target_loader: DataLoader, num_epochs: int = 100, dc_iter: int = 100) -> List[float]:
        self.generator.train()
        self.critic.train()
        losses = []
        source_critic_scores = []
        target_critic_scores = []
        for epoch in trange(num_epochs, desc='Epoch'):
            loss = 0
            for (source_data, _), (target_data, _) in zip(source_loader, target_loader):
                source_data, target_data = source_data.to(self.device), target_data.to(self.device)

                # Train domain critic
                for _ in range(dc_iter):
                    self.critic_optimizer.zero_grad()
                    
                    with torch.no_grad():
                        source_features = self.generator(source_data)
                        target_features = self.generator(target_data)
                    
                    # Compute empirical Wasserstein distance
                    dc_source = self.critic(source_features)
                    dc_target = self.critic(target_features)
                    wasserstein_distance = dc_source.mean() - dc_target.mean()

                    # Gradient penalty
                    gradient_penalty = self.compute_gradient_penalty(source_features, target_features)

                    # Domain critic loss
                    dc_loss = - wasserstein_distance + self.gamma * gradient_penalty
                    dc_loss.backward()
                    self.critic_optimizer.step()

                # Train feature extractor
                self.generator_optimizer.zero_grad()
                source_features = self.generator(source_data)
                target_features = self.generator(target_data)
                dc_source = self.critic(source_features)
                dc_target = self.critic(target_features)
                wasserstein_distance = dc_source.mean() - dc_target.mean()
                wasserstein_distance.backward()
                self.generator_optimizer.step()
                
                with torch.no_grad():
                    loss += wasserstein_distance.item()
                    
            source_critic_scores.append(self.criticize(source_loader.dataset.tensors[0].to(self.device)))
            target_critic_scores.append(self.criticize(target_loader.dataset.tensors[0].to(self.device)))
            losses.append(loss/len(source_loader))
            print(f'\nEpoch {epoch + 1}/{num_epochs} | Loss: {wasserstein_distance.item()}')
        return losses, source_critic_scores, target_critic_scores
    
    @torch.no_grad()
    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        return self.generator(x)
    
    @torch.no_grad()
    def criticize(self, x: torch.Tensor) -> float:
        self.generator.eval()
        self.critic.eval()
        return self.critic(self.generator(x)).mean().item()
    
def gen_data(n: int, d: int, mu: float, delta: List[int]):
    mu = np.full((n, d), mu, dtype=np.float64)
    noise = np.random.normal(loc = 0, scale = 1, size=(n, d))
    X = mu + noise
    labels = np.zeros(n)
    # 5% of the data is abnormal.
    # Anomalies are generated by randomly adding deltas to the data.
    n_anomalies = int(n * 0.05)
    idx = np.random.choice(n, n_anomalies, replace=False)
    if 0 in delta: 
        delta.pop(delta.index(0))
    split_points = sorted(np.random.choice(range(1, len(idx)), len(delta) - 1, replace=False))
    segments = np.split(idx, split_points)
    for i, segment in enumerate(segments):
        X[segment] = X[segment] + delta[i]
    labels[idx] = 1
    return X, labels

def get_next_index(INDEX_FILE):
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            try:
                return int(f.read().strip())  # Read and convert to int
            except ValueError:
                pass  # If file is corrupt, start from 1

    return 1  # Default to 1 if file doesn't exist or is invalid

def update_index(new_index, INDEX_FILE):
    with open(INDEX_FILE, "w") as f:
        f.write(str(new_index))

def update_reference_index(model_name, generator, critic, REFERENCE_FILE):
    with open(REFERENCE_FILE, "a") as f:
        f.write(str(model_name) + "\t"*4 + str(generator) + "\t"*10 + str(critic) + "\n")

def run(input_dim: int, generator_hidden_dims: List[int], critic_hidden_dims: List[int]):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    ns, nt = 1000, 1000
    d = input_dim
    mu_s, mu_t = 0, 2
    delta_s = [0, 1, 2, 3, 4]
    delta_t = [0, 1, 2, 3, 4]
    Xs, Ys = gen_data(ns, d, mu_s, delta_s)
    Xt, Yt = gen_data(nt, d, mu_t, delta_t)

    # Convert to PyTorch tensors
    Xs = torch.FloatTensor(Xs)
    Ys = torch.LongTensor(Ys)
    Xt = torch.FloatTensor(Xt)
    Yt = torch.LongTensor(Yt)
    batch_size = 64
    source_dataset = TensorDataset(Xs, Ys)
    target_dataset = TensorDataset(Xt, Yt)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    model = WDGRL(input_dim=d, generator_hidden_dims=generator_hidden_dims, critic_hidden_dims=critic_hidden_dims)
    num_epochs = 20
    losses, source_critic_scores, target_critic_scores = model.train(source_loader, target_loader, num_epochs=num_epochs, dc_iter=100)

    # Define a dictionary with all the necessary components
    checkpoint = {
        'generator_state_dict': model.generator.state_dict(),
        'critic_state_dict': model.critic.state_dict(),
        'device': model.device,
    }

    MODEL_DIR = "models"  
    INDEX_FILE = os.path.join(MODEL_DIR, "index.txt")
    REFERENCE_FILE = os.path.join(MODEL_DIR, "ref.txt")
    RESULTS_DIR = os.path.join(MODEL_DIR, "results")
    index = get_next_index(INDEX_FILE)
    model_name = f"wdgrl_{index}"
    MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)

    # Save the checkpoint
    torch.save(checkpoint, f"models/{model_name}.pth")
    print(f"Model saved successfully at models/{model_name}.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), losses, 'b-', label='Domain Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Domain Critic Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"models/results/{model_name}/dcl.pdf", format="pdf")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), source_critic_scores, 'b-', label='Source Critic Score')
    plt.plot(range(1, num_epochs+1), target_critic_scores, 'r-', label='Target Critic Score')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Critic Score (Source vs Target)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"models/results/{model_name}/cs.pdf", format="pdf")
    plt.show()
    plt.close()
    update_index(index + 1, INDEX_FILE)
    update_reference_index(model_name, generator_hidden_dims, critic_hidden_dims, REFERENCE_FILE)
    Xt_hat = model.extract_feature(Xt)
    print(Xt_hat[0:10])