import numpy as np
import pandas as pd
import os
import wfdb
from aeon.datasets import load_from_ts_file
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor
from _utils.utils import get_folder_size

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader


class ECGDataset():
    '''
    dataset_name=['af_classification', 'cpsc_2018', 'georgia','noninvasive_fetal_ecg' ,'ptb-xl',  'st_petersburg_incart','chapman_shaoxing', 'cpsc_2018_extra','ningbo','ptb','micmic3_ecg_sub100']
    '''
    def __init__(self, dataset_path='../Data/ECGPretrain',dataset_list=['cpsc_2018','micmic3_ecg_sub100','af_classification',  'georgia','noninvasive_fetal_ecg' ,'ptb-xl',  'st_petersburg_incart','chapman_shaoxing', 'cpsc_2018_extra','ningbo','ptb' ]
                 ,std_freq=100,gen_subject_list_force=0,subject_list_name='ecg_subject_list.csv',sample_same_length=False):
        self.dataset_list = dataset_list
        self.dataset_path =dataset_path
        self.std_freq = std_freq
        self.subject_list_name = subject_list_name
        self.gen_subject_list_force = gen_subject_list_force
        self.sample_same_length = sample_same_length

        subject_list_dict = self.init_subject_list(self.subject_list_name)
        for dataset_name in subject_list_dict:
            subject_list_dict[dataset_name].to_csv((os.path.join(self.dataset_path,dataset_name,subject_list_name)))

        self.subject_list,self.separate_subject_list = self.get_subject_list(self.subject_list_name)
        self.sample_list = self.get_sample_list()

        self._i_data = None
        self.subject_path = None

    def init_subject_list(self, subject_list_name):
        col_name = None
        subject_list_dict = {}        
        for dataset_name in self.dataset_list:
            root_path = os.path.join(self.dataset_path,dataset_name)
            if (not os.path.exists(os.path.join(root_path,subject_list_name))) or self.gen_subject_list_force:
                subject_list = []
                # Traversing the dataset path
                for root, dirs, files in os.walk(root_path):
                    for file in files:
                        file_name,file_type = file.split('.')[0], file.split('.')[-1]
                        if (file_type == 'hea') and (os.path.exists(os.path.join(root, f'{file_name}.dat')) or os.path.exists(os.path.join(root, f'{file_name}.mat'))):
                            subject_path = os.path.relpath(os.path.join(root, file_name), root_path)
                            if '\\' in subject_path:
                                subject_path = subject_path.replace('\\','/')

                            flag, subject_info = self.ecg_is_available(dataset_name,os.path.join(root_path,subject_path))
                            if not flag:
                                continue

                            if subject_path not in subject_list:
                                subject_list.append([dataset_name, subject_path]+list(subject_info.values()))
                                if col_name is None:
                                    col_name = list(subject_info.keys())
                print(f'{os.path.join(root_path,subject_list_name)} saved!')
                # print(subject_list)
                subject_list_dict[dataset_name] = pd.DataFrame(subject_list,columns=['dataset_name','subject_path']+col_name)
                # pd.DataFrame(subject_list,columns=['dataset_name','subject_path']+col_name).to_csv(os.path.join(root_path,subject_list_name))
        return subject_list_dict
    
    def ecg_is_available(self,dataset_name,subject_path):
        subject_data = wfdb.rdrecord(os.path.join(subject_path))
        subject_info = self.get_subject_info(subject_data,dataset_name)
        
        # No ECG signal (abnormal ECG)
        if (subject_info['ecg_n_sig']<=0):
            return False,subject_info
        
        return True,subject_info

    def get_subject_info(self, subject_data,dataset_name):
        ecg_n_sig = len(self.get_ecg_signals(subject_data,dataset_name)[0])

        # n_sig,fs,sig_len,comments,sig_name
        info = {'n_sig':subject_data.n_sig,'ecg_n_sig':ecg_n_sig,'fs':subject_data.fs,'sig_len':subject_data.sig_len,'comments':subject_data.comments,'sig_name':subject_data.sig_name}
        return info
    
    def get_ecg_signals(self, subject_data,dataset_name):
        ecg_sig_name_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V']
        ecg_sig_index = [i  for i,name in enumerate(subject_data.sig_name) if name in ecg_sig_name_list]
        ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'ECG' in name.upper()]
        if dataset_name in ['NIFEADB','ADFECGDB','NIFECGDB','BA-LABOUR']:
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'ABDOMEN' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'DIRECT' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'THORAX' in name.upper()]

        ecg_sig_index = self.dectect_abnormal(subject_data,ecg_sig_index)
        ecg_sig_name = np.array(subject_data.sig_name)[ecg_sig_index]

        return ecg_sig_index,ecg_sig_name

    def dectect_abnormal(self,subject_data,ecg_sig_index):
        p_signal = subject_data.p_signal
        if not isinstance(p_signal, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        
        normal_ecg_signal_index = []
        for i in ecg_sig_index:
            array = p_signal[:,i]
            # Check proportion of missing values
            total_elements = array.size
            nan_count = np.isnan(array).sum()
            nan_ratio = nan_count / total_elements
            if nan_ratio > 0.25:
                continue
            
            # Check whether all values are identical
            clean_array = array[~np.isnan(array)]
            unique_values, counts = np.unique(clean_array, return_counts=True)
            total_elements = clean_array.size
            proportions = counts / total_elements
            exceeding_values = {value: prop for value, prop in zip(unique_values, proportions) if prop > 0.25}
            if exceeding_values:
                continue
            
            normal_ecg_signal_index.append(i)
        
        return normal_ecg_signal_index

    def get_subject_list(self,subject_list_name):
        subject_list = None
        separate_subject_list = {}
        for dataset_name in self.dataset_list:
            dataset_subject_list = pd.read_csv(os.path.join(self.dataset_path,dataset_name,subject_list_name),index_col=0)
            separate_subject_list[dataset_name] = dataset_subject_list
            if subject_list is None:
                subject_list = dataset_subject_list
            else:
                subject_list = pd.concat([subject_list,dataset_subject_list],axis=0, ignore_index=True)
        return subject_list,separate_subject_list

    def get_sample_list(self):
        self.subject_list.loc[:,'n_sample'] = 1
        self.subject_list = self.subject_list.loc[~(self.subject_list['n_sample'].isna() | (self.subject_list['n_sample'] <= 0) | (self.subject_list['n_sample'] == np.inf)),:]
        self.subject_list.loc[:,'sample_num'] = self.subject_list.loc[:,'n_sample']*self.subject_list.loc[:,'ecg_n_sig']

        sample_list = self.subject_list.loc[self.subject_list.index.repeat(self.subject_list["sample_num"].astype(int))].reset_index(drop=True)
        sample_list['sample_sig'] = (sample_list.groupby("subject_path").cumcount())//sample_list['n_sample']
        sample_list['sample_index'] = (sample_list.groupby("subject_path").cumcount())%sample_list['n_sample']
        
        return sample_list

    def __getitem__(self, index):
        dataset_name = self.sample_list.loc[index,'dataset_name']
        subject_path = os.path.join(self.sample_list.loc[index,'dataset_name'],self.sample_list.loc[index,'subject_path'])
        sample_sig = self.sample_list.loc[index,'sample_sig']
        sample_index = self.sample_list.loc[index,'sample_index']
        if self.subject_path != subject_path:
            self.subject_path = subject_path
            self._i_data = wfdb.rdrecord(os.path.join(self.dataset_path,subject_path))
            resample_signal,signal_mean,signal_std,ecg_sig_name = self.process_data(self._i_data,dataset_name)
            self._i_data.resample_signal = resample_signal
            self._i_data.signal_mean = signal_mean
            self._i_data.signal_std = signal_std
            self._i_data.ecg_sig_name = ecg_sig_name
        
        X = self._i_data.resample_signal[:,int(sample_sig)]
        Y = [np.nan]
        return X,Y
    
    def process_data(self, subject_data,dataset_name,ecg_sig_index=None,ecg_sig_name=None):
        if ecg_sig_index is None:
            ecg_sig_index,ecg_sig_name = self.get_ecg_signals(subject_data,dataset_name)

        p_signal = self.linear_interpolation_2d(subject_data.p_signal[:,ecg_sig_index])
        dataset_freq = subject_data.fs
        signal_mean = p_signal.mean(axis=0)
        signal_std = p_signal.std(axis=0)
        p_signal = (p_signal-signal_mean)/signal_std
        resample_signal = resample(p_signal,num=int(p_signal.shape[0]*(self.std_freq/dataset_freq)))
        # print(resample_signal.shape)

        return resample_signal,signal_mean,signal_std,ecg_sig_name

    def linear_interpolation_2d(self,array):
        # Interpolate each column independently
        for j in range(array.shape[1]):
            col = array[:, j]
            nan_indices = np.where(np.isnan(col))[0]
            if nan_indices.size > 0:  # If the current column has NaN
                valid_indices = np.where(~np.isnan(col))[0]
                valid_values = col[~np.isnan(col)]
                col[nan_indices] = np.interp(nan_indices, valid_indices, valid_values)

        return array

    def save2tensor(self,save_path=None,save_name='',batch_size=2**12):
        if save_path == None:
            save_path = os.path.join(self.dataset_path)
        
        if self.sample_same_length:
            # save2tensor bottleneck is torch.concat. so using Dataloader for concat
            tensordata = TenserDataset(self)
            datalodaer = DataLoader(tensordata, batch_size=batch_size,shuffle=False,num_workers=5)
            tensor_X = None
            tensor_Y = None
            for idx,(X,Y) in enumerate(datalodaer):
                print(f"{idx}/{(len(self)//batch_size+1)}:{idx/(len(self)//batch_size+1)*100:.4f}%")
                if tensor_X is None:
                    tensor_X = X
                    tensor_Y =Y
                else:
                    tensor_X = torch.concat([tensor_X,X])
                    tensor_Y = torch.concat([tensor_Y,Y])
            tensor_X = tensor_X.squeeze()
            tensor_Y = tensor_Y.squeeze()
        else:
            tensor_X = None
            tensor_Y = None
            for idx in range(len(self)):
                X,Y = self[idx]
                if idx % 100 == 0:
                    print(f"{idx}/{len(self)}:{idx/(len(self))*100:.4f}%")
                if tensor_X is None:
                    tensor_X = {idx:X}
                    tensor_Y = {idx:Y}
                else:
                    tensor_X[idx] = X
                    tensor_Y[idx] = Y

        torch.save(tensor_X, os.path.join(save_path,f'{save_name}X.pt'))
        torch.save(tensor_Y, os.path.join(save_path,f'{save_name}Y.pt'))

    def __setitem__(self, index, value):
        pass

    def __len__(self):
        return len(self.sample_list)

    def dataset_describe(self):
        describe_dict = {}
        describe_dict['sample'] = len(self.sample_list)
        describe_dict['subject'] = len(self.subject_list)
        total_size = 0
        for dataset_name in self.dataset_list:
            dir_paths = set(list(map(lambda x:os.path.join(*(x.split('/')[:-1])), self.separate_subject_list[dataset_name]['subject_path'])))
            for dir_path in dir_paths:
                total_size += get_folder_size(os.path.join(self.dataset_path,dataset_name,dir_path))
        total_size = total_size/(2**30)
        describe_dict['File_Size'] = total_size
        describe_dict['Channel'] = max(list(map(lambda x:len(x[1:-1].split(', ')),self.sample_list['sig_name'])))
        describe_dict['frequency'] = max(self.sample_list['fs'])
        
        return describe_dict
    

class TenserDataset(Dataset):
    def __init__(self,dataset):
        super().__init__()
        self.dataset =dataset
    
    def __getitem__(self, index):
        X,Y = self.dataset[index]
        return torch.tensor(X).unsqueeze(dim=0).float(),torch.tensor(Y).unsqueeze(dim=0).float()
    
    def __len__(self):
        return len(self.dataset)
