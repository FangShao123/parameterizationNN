import h5py
import torch
import numpy as np
from MyDataSet import MyDataSet
import torch.utils.data as Data
from sklearn.utils import shuffle
# 加载我们处理的数据
def load_our_dataSet(log_flag = False, train_datasets = "train_test.mat", val_datasets = "val_test.mat"):
    if log_flag:
        print("==================load dataSet===============")
        print(train_datasets)
        print(val_datasets)
        print("=============================================")
    # 训练数据集
    f = h5py.File("DataSets/" + train_datasets, 'r')
    x_data = torch.as_tensor(np.array(f['xdata2']), dtype=torch.float32)
    y_data = torch.as_tensor(np.array(f['ydata2']), dtype=torch.float32)
    # 验证数据集
    f = h5py.File("DataSets/" + val_datasets, 'r')
    x_23w = torch.as_tensor(np.array(f['xdata2']), dtype=torch.float32)
    y_23w = torch.as_tensor(np.array(f['ydata2']), dtype=torch.float32)

    train_x = x_data[:3400, :5]
    train_y = y_data[:3400]
    val_x = torch.cat([x_data[3400:, :5], x_23w[:, :5]], dim=0)
    val_y = torch.cat([y_data[3400:], y_23w], dim=0)
    return train_x, train_y, val_x, val_y
# 加载原论文的数据
def load_ori_dataSet(log_flag = False):

    # 4296 数据
    f = h5py.File('DataSets/datasets.mat', 'r')
    x_data = torch.as_tensor(np.array(f['xdata2']), dtype=torch.float32)
    y_data = torch.as_tensor(np.array(f['ydata2']), dtype=torch.float32)
    # 850 数据
    f = h5py.File('DataSets/datasets_S2.mat', 'r')  ## new input based on PP relation
    x_s2 = torch.as_tensor(np.array(f['xdata2']), dtype=torch.float32)
    y_s2 = torch.as_tensor(np.array(f['ydata2']), dtype=torch.float32)
    # 663 数据
    f = h5py.File('DataSets/datasets_23W.mat', 'r')
    x_23w = torch.as_tensor(np.array(f['xdata2']), dtype=torch.float32)
    y_23w = torch.as_tensor(np.array(f['ydata2']), dtype=torch.float32)

    if log_flag:
        print("==================load dataSet===============")
        print('datasets.mat', x_data.size(0))
        print('datasets_S2.mat', x_s2.size(0))
        print('datasets_23W.mat', x_23w.size(0))
        print("=============================================")

    train_x = torch.cat([x_data[:3400, :], x_s2], dim=0)
    train_y = torch.cat([y_data[:3400], y_s2], dim=0)

    val_x = torch.cat([x_data[3400:, :], x_23w], dim=0)
    val_y = torch.cat([y_data[3400:], y_23w], dim=0)

    return train_x, train_y, val_x, val_y

def load_dataSet(log_flag = False, train_datasets = "train_test.mat", val_datasets = "val_test.mat", mix_all = False):
    if train_datasets is None or val_datasets is None:
        train_x, train_y, val_x, val_y = load_ori_dataSet(log_flag)
    else:
        train_x, train_y, val_x, val_y = load_our_dataSet(log_flag, train_datasets, val_datasets)
    # 所有数据是否打乱混合，原比例划分训练验证集
    if mix_all:
        train_len = train_x.size(0)
        all_data_x = torch.cat([train_x, val_x], dim=0).numpy()
        all_data_y = torch.cat([train_y, val_y], dim=0).numpy()
        all_data_x, all_data_y = shuffle(all_data_x, all_data_y, random_state = 0)
        train_x = torch.from_numpy(all_data_x[:train_len])
        val_x = torch.from_numpy(all_data_x[train_len:])
        train_y = torch.from_numpy(all_data_y[:train_len])
        val_y = torch.from_numpy(all_data_y[train_len:])
    print("==================dataSet info===============")
    print("train len: ", len(train_x))
    print("val   len: ", len(val_x))
    print("batch size: ", 4000)
    print("mix   all:", mix_all)
    print("=============================================")
    train_dataset = MyDataSet(train_x, train_y)
    train_db = Data.DataLoader(train_dataset, batch_size=4500, shuffle=True)
    return train_db, val_x, val_y