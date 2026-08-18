import torch
from torch import nn
import torch.nn.functional as F

#Creating network
class DQN(nn.Module):

    #Input, output and hidden layer (defining the layers)
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        #Hidden layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    #forward (calauctions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    # Test the DQN network
    state_dim = 12
    action_dim = 2
    model = DQN(state_dim, action_dim)
    
    # Create sample input
    sample_state = torch.randn(10, state_dim)
    output = model(sample_state)
    
    print(f"Input shape: {sample_state.shape}")
    print(f"Output shape: {output.shape}")



