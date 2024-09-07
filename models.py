import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001, hidden_dim=50, weight_decay=1e-5, dropout=0.2):

        super(LSTM, self).__init__()
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to('cuda:0')
        self.dropout = nn.Dropout(dropout).to('cuda:0')

        self.fc1 = nn.Linear(hidden_dim, 8).to('cuda:0')
        self.dropout2 = nn.Dropout(dropout).to('cuda:0')
        self.flatten = nn.Flatten().to('cuda:0')

        self.fc2 = nn.Linear(8, output_dim).to('cuda:0')

        self.criteria = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


    def forward(self, x):
        x = x.to('cuda:0')

        lstm_out, _ = self.lstm(x)
        lstm_out = F.relu(lstm_out)
        lstm_out = self.dropout(lstm_out)

        output = F.relu(self.fc1(lstm_out))
        output = self.dropout2(output)
        output = self.flatten(output)
        output = F.softmax(self.fc2(output), dim=1)

        return output

    def random_init(self):
        # Randomly initialize weights and biases
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train.tolist()).float().to('cuda:0')
        self.X_test = torch.tensor(self.X_test.tolist()).float().to('cuda:0')
        self.y_train = torch.tensor(self.y_train.tolist()).long().to('cuda:0')
        self.y_test = torch.tensor(self.y_test.tolist()).long()


class Wavenet(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001, hidden_dim=128, weight_decay=1e-5,
                 dropout=0.2, kernel_size=128, padding='causal'):
        super(Wavenet, self).__init__()

        self.cov1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding).to('cuda:0')
        self.maxpool1 = nn.MaxPool1d(2).to('cuda:0')
        self.dropout1 = nn.Dropout(dropout).to('cuda:0')

        self.cov2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=kernel_size//2, padding=padding).to('cuda:0')
        self.maxpool2 = nn.MaxPool1d(2).to('cuda:0')
        self.dropout2 = nn.Dropout(dropout).to('cuda:0')

        self.cov3 = nn.Conv1d(hidden_dim//2, hidden_dim//4, kernel_size=kernel_size//4, padding=padding).to('cuda:0')
        self.maxpool3 = nn.MaxPool1d(2).to('cuda:0')
        self.dropout3 = nn.Dropout(dropout).to('cuda:0')

        self.flat1 = nn.Flatten().to('cuda:0')
        self.fc1 = nn.Linear(hidden_dim//4 * (input_dim[1]//8), hidden_dim//8).to('cuda:0')
        self.dropout4 = nn.Dropout(dropout).to('cuda:0')
        self.fc2 = nn.Linear(hidden_dim//8, output_dim).to('cuda:0')

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.to('cuda:0')
        x = self.F.relu(self.cov1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.F.relu(self.cov2(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.F.relu(self.cov3(x))
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.flat1(x)
        x = self.F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.F.softmax(self.fc2(x))

        return x

    def random_init(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Convert to PyTorch tensors
        self.X_train = torch.tensor(self.X_train.tolist()).float().to('cuda:0')
        self.X_test = torch.tensor(self.X_test.tolist()).float().to('cuda:0')
        self.y_train = torch.tensor(self.y_train.tolist()).long().to('cuda:0')
        self.y_test = torch.tensor(self.y_test.tolist()).long()


    def predict(self, x):
        self.eval()
        x = x.to('cuda:0')
        output = self(x)
        return output


def train_using_optimizer(model, X_train, X_test, y_train, y_test, epoches=200):
    epoch_accuracies = []
    epoch_loss = []
    epoch_test_loss = []

    model.load_data(X_train, X_test, y_train, y_test)

    for epoch in range(epoches):
        model.train()
        model.optimizer.zero_grad()

        output = model(model.X_train)
        loss = model.criteria(output, model.y_train)

        loss.backward()

        model.optimizer.step()

        # Evaluate on the test set
        test_output = model(model.X_test).cpu().detach()
        _, predicted = torch.max(test_output, 1)
        correct = (predicted == model.y_test).sum().item()
        accuracy = correct / len(model.y_test)
        epoch_accuracies.append(accuracy)

        loss_cpu = loss.cpu().detach()
        epoch_loss.append(loss_cpu.item())

        test_loss = model.criteria(test_output, model.y_test).cpu().detach()
        epoch_test_loss.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epoches}], Loss: {loss.item()}, Test Accuracy: {accuracy * 100}%')

    return epoch_accuracies, epoch_loss, epoch_test_loss