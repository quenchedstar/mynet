
# encoding:utf-8
import torch
import torch.utils.data as data
import os
import glob
import os.path
from scipy.io import loadmat
import h5py
import numpy as np

def load_mat_v73(filepath):
    """
    使用 h5py 读取 MATLAB v7.3 格式的 .mat 文件。
    返回一个与 scipy.io.loadmat 兼容的字典。
    """
    data_dict = {}
    with h5py.File(filepath, 'r') as f:
        # MATLAB 保存的变量会成为 HDF5 文件中的数据集
        for k, v in f.items():
            # 检查当前项目 v 是否是一个数据集 (Dataset)
            if isinstance(v, h5py.Dataset):
                # 如果是数据集，才进行读取和转置操作
                data_dict[k] = np.transpose(v[()])
    return data_dict

class DataSet(data.Dataset):

    def __init__(self, B_trans,s_real_trans,TBFs):
        self.B_trans = B_trans
        self.s_real_trans = s_real_trans
        self.TBFs = TBFs

    def __getitem__(self, index):

        return self.B_trans[index],self.s_real_trans[index],self.TBFs[index]

    def __len__(self):
        return self.B_trans.shape[0]

class Validate_DataSet(data.Dataset):

    def __init__(self, B_trans, s_real_trans, TBFs, ActiveVoxSeed, seedvox):
        self.B_trans = B_trans
        self.s_real_trans = s_real_trans
        self.TBFs = TBFs
        self.ActiveVoxSeed = ActiveVoxSeed
        self.seedvox = seedvox

    def __getitem__(self, index):

        return self.B_trans[index],self.s_real_trans[index],self.TBFs[index], self.ActiveVoxSeed[index], self.seedvox[index]

    def __len__(self):
        return self.B_trans.shape[0]

def load_mat_auto(filepath):
    """自动区分 v6/v7.3 的 .mat 读取。

    癫痫 HD-EEG 那批 datayu_*.mat 是 MATLAB 用 '-v7.3' 存的 (单文件上百 MB,
    HDF5 格式), scipy.io.loadmat 会直接抛 NotImplementedError。老的 Localize-MI
    数据是 v6, 用 loadmat 读。这里两条路都留着, 按实际格式自动选。
    """
    try:
        return loadmat(filepath)
    except NotImplementedError:
        return load_mat_v73(filepath)


def get_data_train(load_root, cond, SNRs, batch_size):
    train = os.path.join(load_root, 'train', cond)
    validate = os.path.join(load_root, 'validation', cond)
    for i in range(4):
        tr_data = load_mat_auto(os.path.join(train, f'datayu_{i + 1}.mat'))
        val_data = load_mat_auto(os.path.join(validate, f'datayu_{i + 1}.mat'))
        if i == 0:
            x = tr_data[(''
                         'B_dataStorage')]
            y = tr_data['s_real_dataStorage']
            z = tr_data['TBFs_dataStorage']
            B_tr = x
            s_real_tr = y
            TBFs_tr = z

            m = val_data['B_dataStorage']
            n = val_data['s_real_dataStorage']
            l = val_data['TBFs_dataStorage']
            B_val = m
            s_real_val = n
            TBFs_val = l

        else:
            x = tr_data['B_dataStorage']
            y = tr_data['s_real_dataStorage']
            z = tr_data['TBFs_dataStorage']
            B_tr = np.concatenate((B_tr, x))  # 3-D tensor 的拼接
            s_real_tr = np.concatenate((s_real_tr, y))
            TBFs_tr = np.concatenate((TBFs_tr,z))

            m = val_data['B_dataStorage']
            n = val_data['s_real_dataStorage']
            l = val_data['TBFs_dataStorage']
            B_val = np.concatenate((B_val, m))
            s_real_val = np.concatenate((s_real_val, n))
            TBFs_val = np.concatenate((TBFs_val,l))

    train_data = DataSet(B_tr,s_real_tr,TBFs_tr)
    validate_data = DataSet(B_val,s_real_val,TBFs_val)
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                   pin_memory=False)  # 上面num_workers说是建议在windows上设为0
    valid_loader = data.DataLoader(dataset=validate_data, batch_size=batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)
    return train_loader, valid_loader

def get_data_validation(load_root, cond, SNRs, batch_size):
    file_list = glob.glob(os.path.join(load_root, 'datayu_*.mat')) #找所有datayu_*.mat格式的文件
    file_list.sort()

    if not file_list:
        print(f"警告：在路径 {load_root} 下没有找到任何 'datayu_*.mat' 文件。")
        return None  # 或者可以返回一个空的数据加载器

    b_val_list = []
    s_real_val_list = []
    tbfs_val_list = []
    activevoxseed_list = []
    seedvox_list = []
    print(f"找到了 {len(file_list)} 个验证文件，正在加载...")

    for file_path in file_list:
        val_data = load_mat_auto(file_path)

        b_val_list.append(val_data['B_dataStorage'])
        s_real_val_list.append(val_data['s_real_dataStorage'])
        tbfs_val_list.append(val_data['TBFs_dataStorage'])
        activevoxseed_list.append(val_data['ActiveVoxSeed_dataStorage'][0])
        seedvox_list.append(val_data['seedvox_dataStorage'])

        # 6. 在循环结束后，将列表中的所有数据一次性拼接起来
    B_val = np.concatenate(b_val_list, axis=0)
    s_real_val = np.concatenate(s_real_val_list, axis=0)
    TBFs_val = np.concatenate(tbfs_val_list, axis=0)
    ActiveVoxSeed_val = np.concatenate(activevoxseed_list, axis=0)
    seedvox_val = np.concatenate(seedvox_list, axis=0)


    validate_data = Validate_DataSet(B_val,s_real_val,TBFs_val, ActiveVoxSeed_val, seedvox_val)
    valid_loader = data.DataLoader(dataset=validate_data, batch_size=batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)
    return valid_loader


def get_brainstrom_epilepsy_data(load_root, cond, SNRs):
    train = load_root
    test = load_root
    validate = load_root
    train_data = DataSet(train)
    test_data = DataSet(test)
    validate_data = DataSet(validate)
    return train_data, test_data, validate_data



def get_localizemi_data(load_root, cond, SNRs):
    train = load_root
    test = load_root
    validate = load_root
    train_data = DataSet(train)
    validate_data = DataSet(validate)
    return train_data, validate_data


