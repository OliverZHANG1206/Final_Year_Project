import os
import torch.nn as nn
from sklearn.cluster import SpectralClustering


class NeuralNetwork(nn.Module):
    """
        Description: LSTM Learning Model
            structure:    | Input | LSTM layer1 | LSTM Layer2 | LSTM Layer3 | Linear1 | Linear2 | Linear3 | Softmax |
            size:         |   2   |     256     |     256     |     256     |   500   |   500   |   500   |    5    |
            active func.: |       |    ReLU     |    ReLU     |    ReLU     |  ReLU   |  ReLU   |  ReLU   |         |

        Argument/Parameter:
            Device: Describe running method: GPU or CPU
            dropout: Parameter to avoid over fitting
            lstm_layer: Number of lstm layers
            input_channel: Number of input elements in a tensor eg. [CO2, PIR] -> 2
            output_channel: Number of output elements for each input tensor
            lstm_hidden_layer_num: Number of neuron in one hidden layer
            linear_layer_num: Number of neuron in one fully connected/dense/linear layer
            batch_size = Number of batch/data tensor for training
    """

    def __init__(self, input_dim, lstm_hidden_dim, linear_hidden_dim, output_dim, lstm_layer, timestamps, batch_size, dropout, device):
        super().__init__()
        # Parameter Initialize
        self.device = device
        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        self.timestamps = timestamps
        self.lstm_hidden_dim = lstm_hidden_dim

        # ANN Models Initialize
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layer, dropout=dropout, batch_first=True).to(device)
        self.fully_connect = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_dim, out_features=linear_hidden_dim),
            nn.Linear(in_features=linear_hidden_dim, out_features=linear_hidden_dim),
            nn.Linear(in_features=linear_hidden_dim, out_features=output_dim), nn.ReLU(),
        ).to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)

    def forward(self, input_data):
        out1, (hn, cn) = self.lstm(input_data)
        out1 = out1[:, -1, :]
        output_result = self.fully_connect(out1)
        return output_result

    def weight_init(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)


class ClusteringModel:
    """
        Description: Spectral Clustering Model
        Argument/Parameter:
            model: Spectral clustering Model
            clusters: Number of clusters for classification
            result: Fitting Result
    """

    def __init__(self, clusters):
        self.result = None
        self.clusters = clusters
        self.model = SpectralClustering(affinity='rbf', n_clusters=self.clusters, eigen_solver='arpack',
                                        assign_labels='discretize', n_init=50, eigen_tol=0.0)

    def fit(self, input_data):
        self.result = self.model.fit(input_data)

    def label(self):
        return self.result.labels_
