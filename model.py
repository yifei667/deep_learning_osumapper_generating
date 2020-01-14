import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvLstmNet(nn.Module):
    # model 1
    def __init__(self):
        super(ConvLstmNet, self).__init__()
        self.output_dim = 6
        self.hidden_size = 16
        self.kernel_size = 2
        self.stride = (1,2)

        self.pool = (1, 2)
        self.n0_features = 16
        self.n1_features = 16

        self.n_layers = 1

        self.n0_fc = 64
        self.n1_fc = 64

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.n0_features, self.kernel_size),
            nn.ReLU(),
            nn.Dropout(),

            nn.MaxPool2d(self.pool, stride=self.stride),

            nn.Conv2d(self.n0_features, self.n1_features, self.kernel_size),
            nn.ReLU(),
            nn.Dropout(),

            nn.MaxPool2d(self.pool, stride=self.stride),
        )

        # define the LSTM module
        self.lstm = nn.GRU(567, self.hidden_size,
                            self.n_layers, batch_first=True)
        # intialize the hidden state
        self.init_hidden(1)
        # define the final linear layer
        self.linear1 = nn.Linear(self.hidden_size, self.n0_fc)
        self.linear2 = nn.Linear(self.n0_fc, self.output_dim)

    def forward(self, spec, div):
        mini_batch_size, time_interval, dim0, dim1 = spec.shape
        assert(mini_batch_size == 1)
        spec = spec.view(time_interval, 1, dim0, dim1)

        #print(spec.shape)
        # compute the output of the convolutional layer
        conv_out = self.conv(spec)
        #print(conv_out.shape)

        conv_out = conv_out.view(mini_batch_size, time_interval, -1)

        #print(conv_out.shape, div.shape)

        conv_cat = torch.cat((conv_out, div), dim=2)

        #print(conv_cat.shape)

        #assert(False)

        # compute the output of the lstm layer
        lstm_out, self.hidden = self.lstm(conv_cat)

        #print(lstm_out.shape)

        # extract final output of the lstm layer
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()

        #print("here")

        # compute output of the linear layer
        output = F.relu(self.linear1(lstm_out))
        final_output = torch.tanh(self.linear2(output))

        #print(final_output)

        # return output
        return final_output

    def init_hidden(self, mini_batch_size):
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        self.hidden = self.hidden.to(device)


class ConvLstmNet2(nn.Module):
    # model 2 (momentum, angular momentum)
    def __init__(self):
        super(ConvLstmNet2, self).__init__()
        self.output_dim = 2
        self.hidden_size = 16
        self.kernel_size = 2
        self.stride = (1,2)

        self.pool = (1, 2)
        self.n0_features = 16
        self.n1_features = 16

        self.n_layers = 1

        self.n0_fc = 64
        self.n1_fc = 64

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.n0_features, self.kernel_size),
            nn.ReLU(),
            nn.Dropout(),

            nn.MaxPool2d(self.pool, stride=self.stride),

            nn.Conv2d(self.n0_features, self.n1_features, self.kernel_size),
            nn.ReLU(),
            nn.Dropout(),

            nn.MaxPool2d(self.pool, stride=self.stride),
        )

        # define the LSTM module
        self.lstm = nn.GRU(567, self.hidden_size,
                            self.n_layers, batch_first=True)
        # intialize the hidden state
        self.init_hidden(1)
        # define the final linear layer
        self.linear1 = nn.Linear(self.hidden_size, self.n0_fc)
        self.linear2 = nn.Linear(self.n0_fc, self.output_dim)

    def forward(self, spec, div):
        mini_batch_size, time_interval, dim0, dim1 = spec.shape
        assert(mini_batch_size == 1)
        spec = spec.view(time_interval, 1, dim0, dim1)

        #print(spec.shape)
        # compute the output of the convolutional layer
        conv_out = self.conv(spec)
        #print(conv_out.shape)

        conv_out = conv_out.view(mini_batch_size, time_interval, -1)

        #print(conv_out.shape, div.shape)

        conv_cat = torch.cat((conv_out, div), dim=2)

        #print(conv_cat.shape)

        #assert(False)

        # compute the output of the lstm layer
        lstm_out, self.hidden = self.lstm(conv_cat)

        #print(lstm_out.shape)

        # extract final output of the lstm layer
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()

        #print("here")

        # compute output of the linear layer
        output = F.relu(self.linear1(lstm_out))
        final_output = self.linear2(output)

        #print(final_output)

        # return output
        return final_output

    def init_hidden(self, mini_batch_size):
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        self.hidden = self.hidden.to(device)