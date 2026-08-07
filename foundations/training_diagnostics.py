import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.

        with torch.no_grad():
            stats = []

            for layer in model:
                x = layer(x)
                if isinstance(layer, nn.Linear):

                    mean_val = torch.mean(x).item()
                    std_val = torch.std(x).item()

                    total_neurons= x.shape[-1]
                    dead_neurons = (x <= 0.0).all(dim=0).sum().item()
                    dead_fraction = dead_neurons/total_neurons

                    stats.append({"mean": round(mean_val, 4), "std": round(std_val, 4), "dead_fraction": round(dead_fraction, 4)})

        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        model.zero_grad()
        grad_stats = []

        criterion = nn.MSELoss()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        for layer in model.modules() :

            if isinstance(layer, nn.Linear):
                if layer.weight.grad is not None:
                    g = layer.weight.grad
                    grad_mean = g.mean().item()
                    grad_std = g.std().item()
                    grad_norm = g.norm().item()

                    grad_stats.append({"mean": round(grad_mean,4), "std": round(grad_std, 4), "norm": round(grad_norm, 4)})
        return grad_stats



        

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        for info in activation_stats: 
            for info_grad in gradient_stats:
                if info['dead_fraction'] > 0.5:
                    return 'dead_neurons'
                elif info_grad['norm'] > 1000 or info['std'] > 10.0:
                    return 'exploding_gradients'
                elif info_grad['norm'] < 1e-5 or info['std'] < 0.1:
                    return 'vanishing_gradients'
        return 'healthy'


        
