import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_model(self, file_name='model.pth'):
        model_directory = Path('./models')
        model_directory.mkdir(parents=True, exist_ok=True)
        file_path = model_directory / file_name
        
        torch.save(self.state_dict(), file_path)
    

class Trainer:
    def __init__(self, model, lr, gamma):
        self.learning_rate = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float)
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float)

        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
            reward_tensor = reward_tensor.unsqueeze(0)
            done = (done,)

        predicted_q_values = self.model(state_tensor)
        target_q_values = predicted_q_values.clone()

        for idx in range(len(done)):
            updated_q_value = reward_tensor[idx]
            if not done[idx]:
                updated_q_value += self.gamma * torch.max(self.model(next_state_tensor[idx]))

            target_q_values[idx][torch.argmax(action_tensor[idx]).item()] = updated_q_value

        self.optimizer.zero_grad()
        loss = self.loss_function(target_q_values, predicted_q_values)
        loss.backward()
        self.optimizer.step()
