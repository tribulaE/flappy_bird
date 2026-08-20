import torch
from torch import nn
import torch.nn.functional as F

#Creating network
class DQN(nn.Module):

    #Input, output and hidden layer (defining the layers)
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        #Hidden layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:

            #Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            #Advantages Stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            #Output layer
            self.fc2 = nn.Linear(hidden_dim, action_dim)

    #forward (calauctions)
    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            
            #Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            #Advtanges calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            #Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.fc2(x)

        return Q


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



