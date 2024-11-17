import torch
import torch.nn as nn
import torch.optim as optim


from config.config_params import CONFIG

class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, CONFIG["n_neurons_hl1"])
        self.layer2 = nn.Linear(CONFIG["n_neurons_hl1"], CONFIG["n_neurons_hl2"])
        self.output = nn.Linear(CONFIG["n_neurons_hl2"], output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)


def train_network(model, optimizer, criterion, inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()



def normalize_data_min_max(inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Normalize inputs and targets using Min-Max normalization.

    Parameters:
        inputs (torch.Tensor): Input tensor of shape (batch_size, input_size).
        targets (torch.Tensor): Target tensor of shape (batch_size,).

    Returns:
        inputs_normalized (torch.Tensor): Normalized input tensor.
        targets_normalized (torch.Tensor): Normalized target tensor.
        normalization_params (Dict): Min and max values for reversing normalization.
    """
    # Normalize inputs
    x_min, _ = torch.min(inputs, dim=0)
    x_max, _ = torch.max(inputs, dim=0)
    inputs_normalized = (inputs - x_min) / (x_max - x_min)

    # Normalize targets
    y_min = torch.min(targets)
    y_max = torch.max(targets)
    targets_normalized = (targets - y_min) / (y_max - y_min)

    # Save normalization parameters for reversing
    normalization_params = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    return inputs_normalized, targets_normalized, normalization_params
