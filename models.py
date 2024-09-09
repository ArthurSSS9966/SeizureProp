import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools



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


class CNN1D(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001, hidden_dim=128, weight_decay=1e-5,
                 dropout=0.2, kernel_size=256, padding='same'):
        super(CNN1D, self).__init__()

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
        self.fc1 = nn.Linear((hidden_dim//4 * kernel_size//8), hidden_dim//8).to('cuda:0')
        self.dropout4 = nn.Dropout(dropout).to('cuda:0')
        self.fc2 = nn.Linear(hidden_dim//8, output_dim).to('cuda:0')

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.to('cuda:0')

        # Apply causal padding for the first conv layer
        x = F.relu(self.cov1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # Apply causal padding for the second conv layer
        x = F.relu(self.cov2(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # Apply causal padding for the third conv layer
        x = F.relu(self.cov3(x))
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.flat1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x

    def random_init(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def predict(self, x):
        self.eval()
        x = x.to('cuda:0')
        output = self(x)
        return output


def train_using_optimizer(model, trainloader, valloader, epoches=200, device='cuda:0'):

    for epoch in range(epoches):
        running_loss = 0.0
        model.train()

        for i, inputs in enumerate(trainloader):
            x, y = inputs

            x = x.to(device)
            y = y.long().to(device)

            model.optimizer.zero_grad()

            outputs = model(x)
            outputs = outputs.float()

            loss = model.criteria(outputs, y)

            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:  # Print every 10 mini-batches
                print(f'Epoch [{epoch+1}/{epoches}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

                # Validate the model
                model.eval()

                # Randomly select 100 samples from the validation set
                with torch.no_grad():
                    val_loss = 0.0
                    val_accuracies = []
                    for i, valdata in enumerate(itertools.islice(valloader, 100)):
                        x, y = valdata

                        x = x.to(device)
                        y = y.long().to(device)

                        outputs = model(x)
                        outputs = outputs.float()

                        loss = model.criteria(outputs, y)
                        val_loss += loss.item()

                        accuracy = (torch.argmax(outputs, dim=1) == y).float().mean()
                        val_accuracies.append(accuracy.item())

                    val_accuracies = torch.tensor(val_accuracies)
                    val_accuracies_mean = val_accuracies.mean()
                    print(f'Validation loss: {val_loss / len(valloader)}, Validation accuracy: {val_accuracies_mean}')


    print('Finished Training')

