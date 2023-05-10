import os
import time
import torch
import Export
import SSH_Pi_Conn
import numpy as np
import torch.nn as nn
import Database as db
import Learning_Model as model
import torch.utils.data as Data
import sklearn.preprocessing as preprocess
from tqdm import tqdm
from sklearn.model_selection import KFold

""" Global Parameter """
DATA_START = 0             # Start point in database
DATA_SIZE = 6720           # Total training data size
TIME_PERIOD = 14.0         # Time duration
EPOCH_SIZE = 50           # Epoch for training
BATCH_SIZE = 64            # Batch size
TAGS = 5                   # Classification Result output Tags
INPUT_DIM = 2              # Input features size
TIMESTAMPS = 12            # Time window (Sequence Length)
LSTM_LAYERS = 3            # Numer of LSTM hidden Layers
DECAY_RATE = 1e-6          # Decay Rate for learning
DROPOUT_RATE = 0.2         # Dropout Rate in LSTM to avoid over fitting
LEARNING_RATE = 0.001      # Learning Rate for LSTM
LSTM_HIDDEN_DIM = 256      # Elements in each LSTM Hidden layer
LINEAR_HIDDEN_DIM = 500    # Elements in each Linear layer
best_loss = 100            # Best Loss in Validation set
KFOLD_SIZE = 5
TRAIN_TEST_RATIO = 0.5     # Training ratio

table_name = 'LECTUREHALL'
data_dic = {'CO2': 'CO2 Concentration (ppm)',
            'PIR_TRIGGER': 'PIR_TRIGGER'}
path = {'model': './lstm.pth',
        'result_data': './Results/Result.csv',
        'result_data_total': './Results/Result_Total.csv',
        'result_loss': './Results/Loss.csv',
        'test_accuracy': './Results/test_accuracy.txt',
        'classification': './Results/Classification_Result.png',
        'class-time': './Results/Classification_Time.png',
        'class-train-time': './Results/Classification_Train_Time',
        'train-time': './Results/Train_Time.png',
        'total-time': './Results/Total_Time.png',
        'train': './Results/Training_Loss_Accuracy.png',
        'val': './Results/Validation_Loss_Accuracy.png',
        'learning_result': './Results/Training_Result.png',
        'learning_result_total': './Results/Training_Result_Total.png',
        'standard': './Data/Standard_Value/Classroom/Data.csv'}

""" Static Functions """
def accuracy_calc(predict, actual):
    predict_val = torch.argmax(predict, dim=-1)
    total = len(actual)
    correct = (predict_val == actual).sum().item()
    return correct / total * 100

def numpy2torch(value, float_conv):
    global working_device
    if float_conv:
        tensor_value = torch.from_numpy(value).float().to(working_device)
    else:
        tensor_value = torch.from_numpy(value).to(working_device)
    return tensor_value


""" Core Program """
if __name__ == '__main__':
    # --------- Get Database from Raspberry Pi --------
    SSH_Pi_Conn.get_file_from_Pi()

    # ---------- Pre Check Working Condition ----------
    working_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Creating Models ----------------
    normalize_model = preprocess.MinMaxScaler()
    split_model = KFold(n_splits=KFOLD_SIZE, random_state=None, shuffle=False)
    cluster_model = model.ClusteringModel(clusters=TAGS)
    lstm_model = model.NeuralNetwork(INPUT_DIM, LSTM_HIDDEN_DIM, LINEAR_HIDDEN_DIM, TAGS, LSTM_LAYERS,
                                     timestamps=TIMESTAMPS, batch_size=BATCH_SIZE,
                                     dropout=DROPOUT_RATE, device=working_device)

    # ---------- Optimizer and Loss Function ----------
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY_RATE)
    criterion = nn.CrossEntropyLoss()

    # ------- Preprocessing Data for Clustering -------
    database = db.database()
    # rawdata = database.read_csv(path['standard'])
    rawdata = database.read_csv('./Data/Data_Small_Classroom_Week_1&2.csv')
    # rawdata = database.read_database(table_name)
    datasets = rawdata[list(data_dic.keys())][DATA_START:DATA_START+DATA_SIZE].to_numpy()
    data = normalize_model.fit_transform(datasets)

    # ------ Spectral Clustering Classification -------
    cluster_model.fit(data)
    label = cluster_model.label()
    print("Info: Finish Clustering")

    # ---------- Preprocessing Data for LSTM ----------
    # Split train data and test data
    TRAIN_SIZE = round((DATA_SIZE - TIMESTAMPS) * TRAIN_TEST_RATIO)
    train_val_data, train_val_label, test_data, test_label = database.split_data(data, label, TIMESTAMPS, INPUT_DIM, TRAIN_SIZE)
    test_tensor = Data.TensorDataset(numpy2torch(test_data, float_conv=True), numpy2torch(test_label, float_conv=False))
    test_dataset = Data.DataLoader(dataset=test_tensor, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------- LSTM Training -----------------
    train_total_loss = []      # Recording averaged training loss in each epoch
    train_total_accuracy = []  # Recording averaged training accuracy in each epoch
    val_total_loss = []        # Recording averaged testing loss in each epoch
    val_total_accuracy = []    # Recording averaged testing accuracy in each epoch

    for epoch in range(EPOCH_SIZE):
        train_average_loss = []      # Recording averaged training loss for k fold in each epoch
        train_average_accuracy = []  # Recording averaged training accuracy for k fold in each epoch
        val_average_loss = []        # Recording averaged testing loss for k fold in each epoch
        val_average_accuracy = []    # Recording averaged testing accuracy for k fold in each epoch
        for index, (train_index, val_index) in enumerate(split_model.split(train_val_data, train_val_label)):
            # K Fold
            train_tensor = Data.TensorDataset(numpy2torch(train_val_data[train_index], float_conv=True),
                                              numpy2torch(train_val_label[train_index], float_conv=False))
            val_tensor = Data.TensorDataset(numpy2torch(train_val_data[val_index], float_conv=True),
                                            numpy2torch(train_val_label[val_index], float_conv=False))
            train_dataset = Data.DataLoader(dataset=train_tensor, batch_size=BATCH_SIZE, shuffle=True)
            val_dataset = Data.DataLoader(dataset=val_tensor, batch_size=BATCH_SIZE, shuffle=False)

            # Temp Parameter
            loss_count = []          # Recording loss on each batch
            accuracy_count = []      # Recording accuracy on each batch

            # Training
            # lstm_model.weight_init()
            lstm_model.train()
            train_bar = tqdm(train_dataset)
            for batch_data in train_bar:
                optimizer.zero_grad()
                features, labels = batch_data
                predicted_labels = lstm_model(features)
                loss = criterion(predicted_labels, labels)
                accuracy = accuracy_calc(lstm_model.softmax(predicted_labels), labels)
                loss.backward()
                optimizer.step()

                loss_count.append(loss.item())
                accuracy_count.append(accuracy)
                train_bar.desc = "Training epoch[{}/{}] K-Fold: [{}] loss:{:.3f} acc:{:.2f}".format(epoch + 1,
                                 EPOCH_SIZE, index+1, np.average(np.array(loss_count)), np.average(np.array(accuracy_count)))
            train_average_loss.append(np.average(np.array(loss_count)))
            train_average_accuracy.append(np.average(np.array(accuracy_count)))

            # Evaluation
            loss_count.clear()
            accuracy_count.clear()
            lstm_model.eval()
            with torch.no_grad():
                val_bar = tqdm(val_dataset)
                for batch_data in val_bar:
                    features, labels = batch_data
                    predicted_labels = lstm_model(features)
                    loss = criterion(predicted_labels, labels)
                    accuracy = accuracy_calc(lstm_model.softmax(predicted_labels), labels)

                    loss_count.append(loss.item())
                    accuracy_count.append(accuracy)
                    val_bar.desc = "Testing K-Fold: [{}] loss:{:.3f} acc:{:.2f}".format(
                                   index+1, np.average(np.array(loss_count)), np.average(np.array(accuracy_count)))
            val_average_loss.append(np.average(np.array(loss_count)))
            val_average_accuracy.append(np.average(np.array(accuracy_count)))

        train_total_loss.append(np.average(np.array(train_average_loss)))
        train_total_accuracy.append(np.average(np.array(train_average_accuracy)))
        val_total_loss.append(np.average(np.array(val_average_loss)))
        val_total_accuracy.append(np.average(np.array(val_average_accuracy)))
        print("Train: Average Loss: {:.3f}, Average Accuracy: {:.3f}".format(train_total_loss[epoch], train_total_accuracy[epoch]))
        print("Val: Average Loss: {:.3f}, Average Accuracy: {:.3f}".format(val_total_loss[epoch], val_total_accuracy[epoch]))

        if val_total_loss[epoch] < best_loss:
            best_loss = val_total_loss[epoch]
            torch.save(lstm_model.state_dict(), path['model'])

        time.sleep(0.1)
    print("Finish Training")

    # ----------- Prediction & Testing data -----------
    target_labels = []
    predicted_labels = []
    lstm_model.load_state_dict(torch.load(path['model']))

    lstm_model.eval()
    with torch.no_grad():
        test_bar = tqdm(test_dataset)
        for batch_data in test_bar:
            features, labels = batch_data
            output = lstm_model.softmax(lstm_model(features))
            target_labels.extend(labels.detach().cpu().tolist())
            predicted_labels.extend(torch.argmax(output.detach(), dim=-1).cpu().tolist())

    test_result = ['Testing Accuracy Result: {:.3f} %\n'.format(
        (np.array(predicted_labels) == np.array(target_labels)).sum().item() / len(target_labels) * 100)]
    print("Info: Minimum Loss Would be: {:.3f}".format(best_loss))
    print(test_result[-1])

    # ----------- Export Basic Information ------------
    print("------------ Device Info ------------")
    print(f"Using {working_device} device")
    print("------------ Models Info ------------")
    print('Normalized Model:', normalize_model)
    print('Cluster Model:', cluster_model.model)
    print('Neural Network Model:', lstm_model)
    print('Loss Calculation Method:', criterion)

    # ---------------- Saving Results -----------------
    # Export Figures
    print("----------- Saving Figures ----------")
    Export.export_classification_result(datasets, label, data_dic, path['classification'])
    Export.export_classification_result_with_time(datasets[TIMESTAMPS:, :], label[TIMESTAMPS:], TIME_PERIOD, data_dic, path['class-time'])
    Export.export_classification_result_with_time(datasets[TRAIN_SIZE+TIMESTAMPS+1:, :], label[TRAIN_SIZE+TIMESTAMPS+1:], TIME_PERIOD*(1-TRAIN_TEST_RATIO), data_dic, path['class-train-time'])
    Export.export_learning_process(EPOCH_SIZE, train_total_loss, train_total_accuracy, 'Training', path['train'])
    Export.export_learning_process(EPOCH_SIZE, val_total_loss, val_total_accuracy, 'Validation', path['val'])
    Export.export_learning_result(datasets[TRAIN_SIZE+1:, :], TIMESTAMPS, predicted_labels, target_labels, data_dic, path['learning_result'])
    Export.export_learning_result_with_time(datasets[TRAIN_SIZE+TIMESTAMPS+1:, :], predicted_labels, TIME_PERIOD*(1-TRAIN_TEST_RATIO), data_dic, path['train-time'])
    Export.export_confusion(target_labels, predicted_labels, 5)

    # Export loss and accuracy
    print("----------- Saving Results ----------")
    column_names = ['Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']
    export_data = np.concatenate((np.array(train_total_loss).reshape(EPOCH_SIZE, 1),
                                  np.array(train_total_accuracy).reshape(EPOCH_SIZE, 1),
                                  np.array(val_total_loss).reshape(EPOCH_SIZE, 1),
                                  np.array(val_total_accuracy).reshape(EPOCH_SIZE, 1)), axis=1)
    Export.export_csv(export_data, path['result_loss'], column=column_names)

    # Export data
    column_names = []
    for dataname in data_dic.keys():
        column_names.append('{}_orginal_data'.format(dataname))

    for dataname in data_dic.keys():
        column_names.append('{}_nomalized_data'.format(dataname))

    column_names.extend(['Cluster labels', 'Training labels'])
    export_data = np.concatenate((datasets[TRAIN_SIZE+TIMESTAMPS+1:, :], data[TRAIN_SIZE+TIMESTAMPS+1:, :], np.array(target_labels).reshape(len(target_labels), 1), np.array(predicted_labels).reshape(len(predicted_labels), 1)), axis=1)
    Export.export_csv(export_data, path['result_data'], column=column_names)

    # Export all (training & testing data)
    dataX = np.array([data[index: index + TIMESTAMPS] for index in range(len(data) - TIMESTAMPS)]).reshape(-1, TIMESTAMPS, INPUT_DIM)
    dataY = np.array([label[index + TIMESTAMPS] for index in range(len(label) - TIMESTAMPS)])
    total_tensor = Data.TensorDataset(numpy2torch(dataX, float_conv=True), numpy2torch(dataY, float_conv=False))
    total_dataset = Data.DataLoader(dataset=total_tensor, batch_size=BATCH_SIZE, shuffle=False)

    target_labels = []
    predicted_labels = []

    lstm_model.eval()
    with torch.no_grad():
        total_bar = tqdm(total_dataset)
        for batch_data in total_bar:
            features, labels = batch_data
            output = lstm_model.softmax(lstm_model(features))
            target_labels.extend(labels.detach().cpu().tolist())
            predicted_labels.extend(torch.argmax(output.detach(), dim=-1).cpu().tolist())

    test_result.append('Total Accuracy Result: {:.3f} %\n'.format(
        (np.array(predicted_labels) == np.array(target_labels)).sum().item() / len(target_labels) * 100))
    print(test_result[-1])
    Export.export_learning_result(datasets, TIMESTAMPS, predicted_labels, target_labels, data_dic, path['learning_result_total'])
    Export.export_learning_result_with_time(datasets[TIMESTAMPS:], predicted_labels, TIME_PERIOD, data_dic, path['total-time'])
    export_data = np.concatenate((datasets[TIMESTAMPS:, :], data[TIMESTAMPS:, :], np.array(target_labels).reshape(len(target_labels), 1), np.array(predicted_labels).reshape(len(predicted_labels), 1)), axis=1)
    Export.export_csv(export_data, path['result_data_total'], column=column_names)
    Export.export_txt(test_result, path['test_accuracy'])

    # -------------- Release Memory --------------------
    torch.cuda.empty_cache()
