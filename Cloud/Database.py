import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sqlalchemy import create_engine

class database:
    def __init__(self):
        self.frame = None
        if os.path.exists('./Data/data.db'):
            self.is_connect = True
            self.engine = create_engine('sqlite:///Data/data.db')
            print("Info: Successfully Connected to Database.")
        else:
            self.is_connect = False
            print("Error: Database Not Exist.")

    def read_database(self, table_name):
        if not self.is_connect:
            print("Error: Database Not Connected.")
            return
        else:
            self.frame = pd.read_sql_table(table_name, self.engine)
            return self.frame

    def read_csv(self, path):
        self.frame = pd.read_csv(path)
        return self.frame

    @staticmethod
    def split_data(dataset, labels, timestep, input_dim, train_size):
        # Creating features and labels
        dataX = np.array([dataset[index: index + timestep] for index in range(len(dataset) - timestep)])
        dataY = np.array([labels[index + timestep] for index in range(len(labels) - timestep)])

        # Split Training and Testing dataset
        x_train = dataX[:train_size, :].reshape(-1, timestep, input_dim)
        y_train = dataY[:train_size]
        x_test = dataX[train_size+1:, :].reshape(-1, timestep, input_dim)
        y_test = dataY[train_size+1:]

        return x_train, y_train, x_test, y_test

