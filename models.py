import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from wavenet_modules import DilatedQueue, dilate
import numpy as np



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


class Wavenet(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=4,
                 blocks=1,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=2,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(Wavenet, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        return x


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s


def train_using_optimizer(model, trainloader, valloader, epoches=200, device='cuda:0'):
    for epoch in range(epoches):
        running_loss = 0.0
        model.train()

        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.long().to(device)

            # Zero gradients
            model.optimizer.zero_grad()

            # Forward pass
            outputs = model(x).float()

            # Compute loss and perform backward pass
            loss = model.criteria(outputs, y)
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

            # Print loss every 100 mini-batches
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{epoches}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

                # Validate model every 100 mini-batches
                val_loss, val_accuracy = evaluate_model(model, valloader, device)
                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    print('Finished Training')


def evaluate_model(model, dataloader, device, num_batches=100):
    model.eval()
    val_loss = 0.0
    val_accuracies = []

    with torch.no_grad():
        for i, (x, y) in enumerate(itertools.islice(dataloader, num_batches)):
            x, y = x.to(device), y.long().to(device)

            outputs = model(x).float()
            loss = model.criteria(outputs, y)
            val_loss += loss.item()

            accuracy = (torch.argmax(outputs, dim=1) == y).float().mean()
            val_accuracies.append(accuracy.item())

    val_accuracies_mean = torch.tensor(val_accuracies).mean()
    return val_loss / num_batches, val_accuracies_mean