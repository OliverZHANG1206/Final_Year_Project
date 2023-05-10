import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

COLOR = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])

def export_csv(data, path, column=None):
    dataframe = None
    if type(data) == pd.DataFrame:
        dataframe = data
    if type(data) == np.ndarray:
        dataframe = pd.DataFrame(data.tolist(), columns=column)
    dataframe.to_csv(path, index=False)
    print("Info: Finish Exporting CSV. File Path: {path}".format(path=path))

def export_txt(data, path):
    with open(path, 'w') as f:
        f.writelines(data)

def export_learning_process(epoch_size, loss, accuracy, title, path):
    epoch = [index + 1 for index in range(epoch_size)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10+epoch_size*0.02, 4))
    ax1.plot(epoch, loss)
    ax2.plot(epoch, accuracy)
    if epoch_size <= 80:
        ax1.set_xticks([index * 5 for index in range(int(epoch_size / 5)+1)])
        ax2.set_xticks([index * 5 for index in range(int(epoch_size / 5)+1)])
    else:
        ax1.set_xticks([index * 10 for index in range(int(epoch_size / 10) + 1)])
        ax2.set_xticks([index * 10 for index in range(int(epoch_size / 10) + 1)])
    # ax1.set_yticks([0.1+index*0.1 for index in range(12)])
    ax2.set_yticks([index * 10 for index in range(11)])
    ax1.set(xlabel='Epoch', ylabel='Loss', ylim=(max(0, min(loss)-0.05), max(loss)+0.05), title='{title} Loss'.format(title=title))
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 100), title='{title} Accuracy'.format(title=title))
    # plt.show()
    fig.savefig(path)
    print("Info: Finish Exporting Picture. File Path: {path}".format(path=path))

def export_learning_result(datasets, timestamps, predict, target, data_dic, path, color=COLOR):
    fig = None
    data_list = list(data_dic.keys())
    data_label = list(data_dic.values())
    color_tags1 = [color[predict[index]] for index in range(len(predict))]
    color_tags2 = [color[target[index]] for index in range(len(target))]

    if len(data_list) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.scatter(x=datasets[timestamps:, 0], y=datasets[timestamps:, 1], s=80, marker=".", c=color_tags1)
        ax2.scatter(x=datasets[timestamps:, 0], y=datasets[timestamps:, 1], s=80, marker=".", c=color_tags2)
        ax1.set(xlabel=data_label[0], ylabel=data_label[1], title='Training Result')
        ax2.set(xlabel=data_label[0], ylabel=data_label[1], title='Classification')
    elif len(data_list) == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={"projection": "3d"})
        ax1.scatter(xs=datasets[timestamps:, 0], ys=datasets[timestamps:, 1], zs=datasets[timestamps:, 2],
                    s=80, marker=".", c=color_tags1)
        ax2.scatter(xs=datasets[timestamps:, 0], ys=datasets[timestamps:, 1], zs=datasets[timestamps:, 2],
                    s=80, marker=".", c=color_tags2)
        ax1.set(xlabel=data_label[0], ylabel=data_label[1], zlabel=data_label[2])
        ax2.set(xlabel=data_label[0], ylabel=data_label[1], zlabel=data_label[2])
    # plt.show()
    fig.savefig(path)
    print("Info: Finish Exporting Picture. File Path: {path}".format(path=path))

def export_learning_result_with_time(datasets, label, day, data_dic, path, color=COLOR):
    data_label = list(data_dic.values())
    time = [index * day / datasets.shape[0] for index in range(datasets.shape[0])]
    color_tags = [color[label[index]] for index in range(len(label))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.scatter(x=time, y=datasets[:, 0], ls='--', s=30, marker=".", c=color_tags)
    ax2.scatter(x=time, y=datasets[:, 1], ls='--', s=30, marker=".", c=color_tags)
    ax1.set(xlabel='Time (days)', ylabel=data_label[0], title='Training Result')
    ax2.set(xlabel='Time (days)', ylabel=data_label[1])
    fig.savefig(path)
    # plt.show()
    print("Info: Finish Exporting Picture. File Path: {path}".format(path=path))

def export_classification_result(datasets, label, data_dic, path, color=COLOR):
    fig = None
    data_list = list(data_dic.keys())
    data_label = list(data_dic.values())
    color_tags = [color[label[index]] for index in range(len(label))]
    if len(data_list) == 2:
        fig, ax = plt.subplots()
        ax.scatter(x=datasets[:, 0], y=datasets[:, 1], s=60, marker=".", c=color_tags)
        ax.set(xlabel=data_label[0], ylabel=data_label[1], title='Classification')
    elif len(data_list) == 3:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(xs=datasets[:, 0], ys=datasets[:, 1], zs=datasets[:, 2], s=60, marker=".", c=color_tags)
        ax.set(xlabel=data_label[0], ylabel=data_label[1], zlabel=data_label[2], title='Classification')
    fig.savefig(path)
    print("Info: Finish Exporting Picture. File Path: {path}".format(path=path))


def export_classification_result_with_time(datasets, label, day, data_dic, path, color=COLOR):
    data_label = list(data_dic.values())
    time = [index * day / datasets.shape[0] for index in range(datasets.shape[0])]
    color_tags = [color[label[index]] for index in range(len(label))]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.scatter(x=time, y=datasets[:, 0], ls='--', s=30, marker=".", c=color_tags)
    ax2.scatter(x=time, y=datasets[:, 1], ls='--', s=30, marker=".", c=color_tags)
    ax1.set(xlabel='Time (days)', ylabel=data_label[0], title='Classification')
    ax2.set(xlabel='Time (days)', ylabel=data_label[1])
    fig.savefig(path)
    # plt.show()
    print("Info: Finish Exporting Picture. File Path: {path}".format(path=path))

def export_confusion(predict, target, cluster):
    confusion = confusion_matrix(target, predict)
    #plt.colorbar()

    # Add axis labels
    #tick_marks = np.arange(5)
    #plt.xticks(tick_marks, ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    #plt.yticks(tick_marks, ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    #plt.xlabel('Predicted')
    #plt.ylabel('Actual')

    # Add counts to each cell
    #thresh = confusion.max() / 2.
    #for i, j in np.ndindex(confusion.shape):
    #    plt.text(j, i, int(confusion[i, j]),
    #             horizontalalignment="center",
    #             color="white" if confusion[i, j] > thresh else "black")

