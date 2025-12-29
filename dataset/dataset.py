
import random
import numpy as np
import pandas as pd
import os
import wfdb
from aeon.datasets import load_from_ts_file
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor
from utils.utils import get_folder_size
from dataset.ECGDataset import ECGDataset
import torch


class PretrainDataset(ECGDataset):
    '''
    dataset_name=['af_classification', 'cpsc_2018', 'georgia','noninvasive_fetal_ecg' ,'ptb-xl',  'st_petersburg_incart',
    'chapman_shaoxing', 'cpsc_2018_extra','ningbo','ptb','micmic3_ecg_sub100']
    '''
    def __init__(self, dataset_path='../Data/ECGPretrain',dataset_list=['cpsc_2018','af_classification',  'mimic3_ecg_sub0001',
                                                                                      'georgia','noninvasive_fetal_ecg' ,'ptb-xl',  'st_petersburg_incart','chapman_shaoxing', 'cpsc_2018_extra','ningbo','ptb' ]
                 ,std_freq=100,sample_length=500,gen_subject_list_force=0,subject_list_name='pretrain_sample_list.csv',sample_same_length=True):
        self.sample_length =sample_length
        super().__init__(dataset_path=dataset_path,dataset_list=dataset_list,std_freq=std_freq,gen_subject_list_force=gen_subject_list_force,subject_list_name=subject_list_name,sample_same_length=sample_same_length)
        
    
    def ecg_is_available(self,dataset_name,subject_path):
        subject_data = wfdb.rdrecord(subject_path)
        subject_info = self.get_subject_info(subject_data,dataset_name=dataset_name)
        
        # No ECG signal (abnormal) or insufficient signal length
        if (subject_info['ecg_n_sig']<=0) or ((subject_info['sig_len']/subject_info['fs']*self.std_freq)//self.sample_length)<=0:
            return False,subject_info
        
        return True,subject_info

    def get_sample_list(self,):
        self.subject_list.loc[:,'n_sample'] = (self.subject_list['sig_len']/self.subject_list['fs']*self.std_freq)//self.sample_length
        self.subject_list = self.subject_list.loc[~(self.subject_list['ecg_n_sig'] <=0 |self.subject_list['n_sample'].isna() | (self.subject_list['n_sample'] <= 0) | (self.subject_list['n_sample'] == np.inf)),:]
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
            resample_signal,signal_mean,signal_std,ecg_sig_name = self.process_data(self._i_data,dataset_name=dataset_name)
            self._i_data.resample_signal = resample_signal
            self._i_data.signal_mean = signal_mean
            self._i_data.signal_std = signal_std
            self._i_data.ecg_sig_name = ecg_sig_name
        try:
            X = self._i_data.resample_signal[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1)),int(sample_sig)]
            Y = [np.nan]
            return X,Y
        except:
            print(f"{index}/{dataset_name}/{subject_path}/{sample_sig}/{sample_index}")

    def save2tensor_parallel(self, save_path=None, save_name=''):
        if save_path is None:
            save_path = os.path.join(self.dataset_path)

        min_idx_df = self.sample_list[['subject_path']].reset_index().groupby('subject_path').min()
        max_idx_df = self.sample_list[['subject_path']].reset_index().groupby('subject_path').max()


        def process(subject_path):
            min_idx = min_idx_df.loc[subject_path, 'index']
            max_idx = max_idx_df.loc[subject_path, 'index']

            _i_data = wfdb.rdrecord(os.path.join(self.dataset_path,os.path.join(self.sample_list.loc[min_idx,'dataset_name'],self.sample_list.loc[min_idx,'subject_path'])))
            resample_signal,signal_mean,signal_std,ecg_sig_name = self.process_data(_i_data)
            _i_data.resample_signal = resample_signal

            print(f"read {subject_path}/{len(min_idx_df)},{min_idx}-{max_idx}")
            tensor_X = None
            tensor_Y = None
            for index in range(min_idx,max_idx+1):
                sample_sig = self.sample_list.loc[index,'sample_sig']
                sample_index = self.sample_list.loc[index,'sample_index']
                X = _i_data.resample_signal[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1)),int(sample_sig)]
                Y = [np.nan]

                X = torch.tensor(X).unsqueeze(dim=0).float()
                Y = torch.tensor(Y).unsqueeze(dim=0).float()

                if tensor_X is None:
                    if self.sample_same_length:
                        tensor_X = X
                        tensor_Y = Y
                    else:
                        tensor_X = {idx:X}
                        tensor_Y = {idx:Y}
                else:
                    if self.sample_same_length:
                        tensor_X = torch.concat([tensor_X,X])
                        tensor_Y = torch.concat([tensor_Y,Y])
                    else:
                        tensor_X[idx] = X
                        tensor_Y[idx] = Y
            return list(range(min_idx,max_idx+1)),tensor_X,tensor_Y

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process, list(self.sample_list[['subject_path']].groupby('subject_path').min().index)))

        # Now we have the results for all indices in the form of (X, Y) tuples

        tensor_X = None
        tensor_Y = None
        indices = None
        for idx, (index ,X, Y) in enumerate(results):
            
            if tensor_X is None:
                indices = index
                if self.sample_same_length:
                    tensor_X = X
                    tensor_Y = Y
                else:
                    tensor_X = X
                    tensor_Y = Y
            else:
                indices += index
                if self.sample_same_length:
                    tensor_X = torch.concat([tensor_X, X])
                    tensor_Y = torch.concat([tensor_Y, Y])
                else:
                    tensor_X.update(X)
                    tensor_Y.update(Y)
        # make tensor_X index is same to samle_list index
        indices = torch.tensor(indices).sort().indices
        # print((indices),tensor_X.shape)
        tensor_X = tensor_X[indices,:]
        tensor_Y = tensor_Y[indices,:]
        torch.save(tensor_X, os.path.join(save_path, f'{save_name}X.pt'))
        torch.save(tensor_Y, os.path.join(save_path, f'{save_name}Y.pt'))


class ClassificationDataset(ECGDataset):
    '''
    dataset_path='../Data/PhysioNet_in_Cardiology_Challenge'
    dataset_name=['cpsc_2018','Reducing_False_Arrhythmia_Alarms_2015']
    '''
    def __init__(self, dataset_path='../Data/ECGPretrain',dataset_list=['cpsc_2018']
                 ,std_freq=100,sample_length=500,gen_subject_list_force=1,
                 subject_list_name='classification_subject_list.csv',sample_same_length=True
                 ,undersample=False):
        self.class_dict = {}
        self.sample_length =sample_length
        self.dataset_name = dataset_list[0]
        
        super().__init__(dataset_path=dataset_path,dataset_list=dataset_list,std_freq=std_freq,gen_subject_list_force=gen_subject_list_force,subject_list_name=subject_list_name,sample_same_length=sample_same_length)
        if undersample:
            subject_list = self.undersample()
            self.raw_subject_list = self.subject_list.copy()
            self.subject_list = subject_list
        self.num_class = len(set(self.subject_list['class']))
        self.sample_list = self.get_sample_list()

    def ecg_is_available(self,dataset_name,subject_path):
        subject_data = wfdb.rdrecord(os.path.join(subject_path))
        subject_info = self.get_subject_info(subject_data,dataset_name)
        Y = self.get_class(subject_data,dataset_name=dataset_name,subject_path=subject_path)

        # No ECG signal (abnormal) or insufficient signal length
        if (subject_info['ecg_n_sig']<=0) or ((subject_info['sig_len']/subject_info['fs']*self.std_freq)//self.sample_length)<=0 or (Y is None):
            return False,subject_info
        
        subject_info['class'] = Y[0]
        return True,subject_info

    def get_sample_list(self,):
        self.subject_list.loc[:,'n_sample'] = (self.subject_list['sig_len']/self.subject_list['fs']*self.std_freq)//self.sample_length
        self.subject_list = self.subject_list.loc[~(self.subject_list['ecg_n_sig'] <=0 |self.subject_list['n_sample'].isna() | (self.subject_list['n_sample'] <= 0) | (self.subject_list['n_sample'] == np.inf)),:]
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
            self._i_data.subject_path = subject_path
            self._i_data.dataset_name = self.sample_list.loc[index,'dataset_name']
            self._i_data.resample_signal = resample_signal
            self._i_data.signal_mean = signal_mean
            self._i_data.signal_std = signal_std
            self._i_data.ecg_sig_name = ecg_sig_name
            if self.undersample:
                self._i_data.Y = self.sample_list.loc[index,'class']
            else:
                self._i_data.Y = self.get_class(self._i_data,self.sample_list.loc[index,'dataset_name'],subject_path)
        
        X = self._i_data.resample_signal[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1)),int(sample_sig)]
        Y = self._i_data.Y
        return X,Y

    def get_class(self,data,dataset_name,subject_path):
        Y = []
        if dataset_name == 'cpsc_2018':
            Dx = [comment.split(':')[-1].split(',')  for comment in data.comments if 'Dx' in comment][0]
            if len(Dx)>=2:
                return None
            else:
                if Dx[0] not in self.class_dict:
                    self.class_dict[Dx[0]] = len(self.class_dict)
                Y.append(self.class_dict[Dx[0]])

        elif dataset_name == 'mimic_perform_af':
            if 'non_af' in subject_path:
                if 'non_af' not in self.class_dict:
                    self.class_dict['non_af'] = 0
                Y.append(self.class_dict['non_af'])
            else:
                if 'af' not in self.class_dict:
                    self.class_dict['af'] = 1
                Y.append(self.class_dict['af'])
        elif dataset_name == 'cpsc_2021':
            subjuet_descrip = ';'.join(data.comments)

            if 'non atrial fibrillation' in subjuet_descrip:
                if 'non atrial fibrillation' not in self.class_dict:
                    self.class_dict['non atrial fibrillation'] = 0
                Y.append(self.class_dict['non atrial fibrillation'])
            elif 'persistent atrial fibrillation' in subjuet_descrip:
                if 'persistent atrial fibrillation' not in self.class_dict:
                    self.class_dict['persistent atrial fibrillation'] = 1
                Y.append(self.class_dict['persistent atrial fibrillation'])
            else:
                if 'paroxysmal atrial fibrillation' not in self.class_dict:
                    self.class_dict['paroxysmal atrial fibrillation'] = 2
                Y.append(self.class_dict['paroxysmal atrial fibrillation'])
        elif dataset_name == 'Reducing_False_Arrhythmia_Alarms_2015':
            alarm_type, alarm_label= data.comments
            alarm_label = False if alarm_label[0]=='F' else True
            label = alarm_type if alarm_label else 0
            if label not in self.class_dict:
                self.class_dict[label] = len(self.class_dict)
            Y.append(self.class_dict[label])
        elif dataset_name == 'ptb':
            Dx = [comment.split(':')[-1].split(',')  for comment in data.comments if 'Dx' in comment][0]
            if len(Dx)>=2:
                return None
            else:
                if Dx[0] not in self.class_dict:
                    self.class_dict[Dx[0]] = len(self.class_dict)
                Y.append(self.class_dict[Dx[0]])
        elif dataset_name == 'st_petersburg_incart':
            Dx = [comment.split(':')[-1].split(',')  for comment in data.comments if 'Dx' in comment][0]
            if len(Dx)>=2:
                return None
            else:
                if Dx[0] not in self.class_dict:
                    self.class_dict[Dx[0]] = len(self.class_dict)
                Y.append(self.class_dict[Dx[0]])
        elif dataset_name == 'georgia':
            Dx = [comment.split(':')[-1].split(',')  for comment in data.comments if 'Dx' in comment][0]
            if len(Dx)>=2:
                return None
            else:
                if Dx[0] not in self.class_dict:
                    self.class_dict[Dx[0]] = len(self.class_dict)
                Y.append(self.class_dict[Dx[0]])
        elif dataset_name == 'NIFEADB':
            if 'ARR' in subject_path:
                if 'ARR' not in self.class_dict:
                    self.class_dict['ARR'] = 0
                Y.append(self.class_dict['ARR'])
            else:
                if 'NR' not in self.class_dict:
                    self.class_dict['NR'] = 1
                Y.append(self.class_dict['NR'])
        return Y

    def get_class_sample(self,beta=2.01):
        self.subject_list['num_sample'] = self.subject_list['ecg_n_sig']/self.subject_list['fs']*self.std_freq*self.subject_list['sig_len']/self.sample_length
        class_sample = self.subject_list[['class','num_sample']].groupby('class').sum().sort_values('num_sample')
        for i, c in enumerate(class_sample.index):
            res = 0
            for j in range(i+1,len(class_sample)):
                res += class_sample.iloc[j]['num_sample'] if (beta*(class_sample.iloc[i]['num_sample'])) >= class_sample.iloc[j]['num_sample'] else beta*(class_sample.iloc[i]['num_sample'])
            class_sample.loc[c,'total_sample'] = res
            class_sample.loc[c,'score'] = res*(len(class_sample)-i)

        return class_sample
    def undersample(self, beta=2.01):
        self.class_sample = self.get_class_sample()
        subject_list = []
        if self.dataset_name == 'cpsc_2018':
            min_num_sample_class = 4
            drop_class = [7,8]
        elif self.dataset_name == 'ptb':
            min_num_sample_class = 6
            drop_class = [8,9]
        elif self.dataset_name == 'st_petersburg_incart':
            min_num_sample_class = 0
            drop_class = []
        elif self.dataset_name == 'georgia':
            min_num_sample_class = 24
            drop_class = [25, 34, 37, 27, 30, 36, 29, 16, 33, 35, 32, 28, 20, 31, 23, 19, 13]
        elif self.dataset_name == 'cpsc_2021':
            min_num_sample_class = 2
            drop_class = []
        elif self.dataset_name == 'mimic_perform_af':
            min_num_sample_class = 0
            drop_class = []
        elif self.dataset_name == 'Reducing_False_Arrhythmia_Alarms_2015':
            min_num_sample_class = 3
            drop_class = [4,5]
        else:
            min_num_sample_class = 0
            drop_class = []

        undersample_class = set(self.class_sample.index)-set(drop_class)
        min_num_sample = self.class_sample.loc[min_num_sample_class,'num_sample']
        dataset_sample = self.subject_list[self.subject_list['dataset_name']==self.dataset_name]
        dataset_sample = dataset_sample.drop(dataset_sample.index[dataset_sample['class'].isin(drop_class)])

        for c in undersample_class:
            class_sample = dataset_sample[dataset_sample['class'] == c]
            while class_sample['num_sample'].sum()>(beta*min_num_sample):
                class_sample = class_sample.drop(index=random.choice(class_sample.index))
                class_sample = class_sample.reset_index(drop=True)
            subject_list.append(class_sample)
        subject_list = pd.concat(subject_list).reset_index()
        subject_list['class'] = subject_list['class'].replace({c:i for i,c in enumerate(undersample_class)})

        return subject_list
        

    def dataset_describe(self):
        describe_dict = super().dataset_describe()
        describe_dict['num_class'] = self.num_class


        return describe_dict


class DetectionDataset(ECGDataset):
    '''
    '''
    def __init__(self, dataset_path='../Data/PhysioNet_in_Cardiology_Challenge',dataset_list=['Noninvasive_Fetal_ECG_2013']
                 ,std_freq=100,sample_length=500,gen_subject_list_force=0,subject_list_name='detection_subject_list.csv',sample_same_length=True):
        self.sample_length =sample_length
        super().__init__(dataset_path=dataset_path,dataset_list=dataset_list,std_freq=std_freq,gen_subject_list_force=gen_subject_list_force,subject_list_name=subject_list_name,sample_same_length=sample_same_length)
        
    
    def ecg_is_available(self,dataset_name,subject_path):
        subject_data = wfdb.rdrecord(os.path.join(subject_path))
        subject_info = self.get_subject_info(subject_data,dataset_name)
        Y,ann_sample = self.get_Y(subject_data,dataset_name,subject_path)
        # print(ann_sample)
        # No ECG signal (abnormal) or insufficient signal length
        if (subject_info['ecg_n_sig']<=0) or ((subject_info['sig_len']/subject_info['fs']*self.std_freq)//self.sample_length)<=0 or (Y is None):
            return False,subject_info
        if ann_sample[0]/subject_data.fs*self.std_freq>self.sample_length:
            subject_info['begin'] = ann_sample[0]
        else:
            subject_info['begin'] = -1
        if (subject_data.sig_len - ann_sample[-1])/subject_data.fs*self.std_freq>self.sample_length:
            subject_info['end'] = ann_sample[-1]
        else:
            subject_info['end'] = subject_data.sig_len

        return True,subject_info

    def get_sample_list(self,):
        self.subject_list.loc[:,'n_sample'] = ((self.subject_list['end']-self.subject_list['begin']-1)/self.subject_list['fs']*self.std_freq)//self.sample_length
        self.subject_list = self.subject_list.loc[~(self.subject_list['ecg_n_sig'] <=0 |self.subject_list['n_sample'].isna() | (self.subject_list['n_sample'] <= 0) | (self.subject_list['n_sample'] == np.inf)),:]
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
            begin_point = self.sample_list.loc[index,'begin']
            end_point = self.sample_list.loc[index,'end']
            self._i_data.p_signal = self._i_data.p_signal[begin_point+1:end_point,:]
            resample_signal,signal_mean,signal_std,ecg_sig_name = self.process_data(self._i_data,dataset_name)
            self._i_data.subject_path = subject_path
            self._i_data.dataset_name = self.sample_list.loc[index,'dataset_name']
            self._i_data.resample_signal = resample_signal
            self._i_data.signal_mean = signal_mean
            self._i_data.signal_std = signal_std
            self._i_data.ecg_sig_name = ecg_sig_name
            self._i_data.Y,_ = self.get_Y(self._i_data,self.sample_list.loc[index,'dataset_name'],os.path.join(self.dataset_path,subject_path),begin_point,end_point)
        
        X = self._i_data.resample_signal[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1)),int(sample_sig)]
        Y = self._i_data.Y[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1))]
        return X,Y

    def get_Y(self,data,dataset_name,subject_path, begin_point=-1,end_point=None):
        Y = []
        ann_sample = None
        if end_point is None:
            end_point = data.sig_len
        if dataset_name == 'Noninvasive_Fetal_ECG_2013':
            if not os.path.exists(subject_path+'.fqrs'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'fqrs')
            Y = np.zeros(int(data.sig_len/data.fs*self.std_freq))
            ann_sample = subject_annotation.sample.astype(int)-1
        elif dataset_name == 'Detection_of_Heart_Beats_2014':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample.astype(int)-1
        elif dataset_name == 'cpsc_2019':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample.astype(int)-1
        elif dataset_name == 'EDB':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'INCART_DB':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'MITDB':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        # elif dataset_name == 'QTDB':
        #     if not os.path.exists(subject_path+'.pu'):
        #         return None,new_sample
        #     subject_annotation = wfdb.rdann(subject_path, 'pu')
        #     ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'SVDB':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'ADFECGDB':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'NIFECGDB':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'BA-LABOUR':
            if not os.path.exists(subject_path+'_Fetal.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path+'_Fetal', 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='N'].copy()-1
        elif dataset_name == 'MITPDB':
            if not os.path.exists(subject_path+'.pwave'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'pwave')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='p'].copy()-1
        elif dataset_name == 'cpsc_2020':
            if not os.path.exists(subject_path+'.atr'):
                return None,ann_sample
            subject_annotation = wfdb.rdann(subject_path, 'atr')
            ann_sample = subject_annotation.sample[np.array(subject_annotation.symbol)=='V'].copy()-1
        
        if len(ann_sample)==0:
            return None,ann_sample
        
        Y = np.zeros(int(data.sig_len))
        Y[ann_sample] = 1
        Y = Y[begin_point+1:end_point]
        
        new_sample = (np.where(Y == 1)[0]/data.fs*self.std_freq).astype(int)
        # print(new_sample)
        Y = np.zeros(int(len(Y)/data.fs*self.std_freq))
        Y[new_sample] = 1
        
        return Y,ann_sample


class ForecastDataset(ECGDataset):
    '''
    dataset_name=['af_classification', 'cpsc_2018', 'georgia','noninvasive_fetal_ecg' ,'ptb-xl',  'st_petersburg_incart',
    'chapman_shaoxing', 'cpsc_2018_extra','ningbo','ptb','micmic3_ecg_sub100']
    '''
    def __init__(self, dataset_path='../Data/ECGPretrain',dataset_list=['cpsc_2018','micmic3_ecg_sub100','af_classification',  
                                                                                      'georgia','noninvasive_fetal_ecg' ,'ptb-xl',  'st_petersburg_incart','chapman_shaoxing', 'cpsc_2018_extra','ningbo','ptb' ]
                 ,std_freq=100,sample_length=500,predict_length=3000,gen_subject_list_force=0,subject_list_name='forecast_subject_list.csv',sample_same_length=True):
        self.sample_length =sample_length
        self.predict_length = predict_length
        super().__init__(dataset_path=dataset_path,dataset_list=dataset_list,std_freq=std_freq,gen_subject_list_force=gen_subject_list_force,subject_list_name=f'{predict_length}_{subject_list_name}',sample_same_length=sample_same_length)
        
    
    def ecg_is_available(self,dataset_name,subject_path):
        subject_data = wfdb.rdrecord(subject_path)
        subject_info = self.get_subject_info(subject_data,dataset_name)

        # No ECG signal (abnormal) or insufficient signal length
        if (subject_info['ecg_n_sig']<=0) or ((subject_info['sig_len']/subject_info['fs']*self.std_freq-self.predict_length)//self.sample_length)<=0:
            return False,subject_info
        
        return True,subject_info

    def get_sample_list(self,):
        self.subject_list.loc[:,'n_sample'] = (self.subject_list['sig_len']/self.subject_list['fs']*self.std_freq-self.predict_length)//self.sample_length
        self.subject_list = self.subject_list.loc[~(self.subject_list['ecg_n_sig'] <=0 |self.subject_list['n_sample'].isna() | (self.subject_list['n_sample'] <= 0) | (self.subject_list['n_sample'] == np.inf)),:]
        self.subject_list.loc[:,'sample_num'] = self.subject_list.loc[:,'n_sample']*self.subject_list.loc[:,'ecg_n_sig']

        sample_list = self.subject_list.loc[self.subject_list.index.repeat(self.subject_list["sample_num"].astype(int))].reset_index(drop=True)
        sample_list['sample_sig'] = (sample_list.groupby("subject_path").cumcount())//sample_list['n_sample']
        sample_list['sample_index'] = (sample_list.groupby("subject_path").cumcount())%sample_list['n_sample']
        
        return sample_list

    def __getitem__(self, index):
        dataset_name=self.sample_list.loc[index,'dataset_name']
        subject_path = os.path.join(dataset_name,self.sample_list.loc[index,'subject_path'])
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
        
        X = self._i_data.resample_signal[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1)),int(sample_sig)]
        Y = self._i_data.resample_signal[int(self.sample_length*(sample_index+1)):(int(self.sample_length*(sample_index+1))+self.predict_length),int(sample_sig)]
        return X,Y


class GenerationDataset(ECGDataset):
    def __init__(self, dataset_path='../Data/PhysioNet_in_Cardiology_Challenge',sub_dataset_name=None,dataset_list=['Noninvasive_Fetal_ECG_2013']
                 ,std_freq=100,sample_length=500,gen_subject_list_force=0,subject_list_name='generation_subject_list.csv',sample_same_length=True):
        self.sample_length =sample_length
        self.sub_dataset_name = sub_dataset_name
        super().__init__(dataset_path=dataset_path,dataset_list=dataset_list,std_freq=std_freq,gen_subject_list_force=gen_subject_list_force,subject_list_name=subject_list_name,sample_same_length=sample_same_length)
        self.noise_dict = {}
    
    def ecg_is_available(self,dataset_name,subject_path):
        subject_data = wfdb.rdrecord(os.path.join(subject_path))
        subject_info = self.get_subject_info(subject_data,dataset_name)
        if dataset_name in ['MITDB','ptb-xl']:
            subject_info['ecg_n_sig'] = 6
        Y = self.get_Y(subject_data,dataset_name,subject_path)
        # print(subject_info['sig_len']/subject_info['fs']*self.std_freq)
        # print(ann_sample)
        # No ECG signal (abnormal) or insufficient signal length
        if (subject_info['ecg_n_sig']<=0) or ((subject_info['sig_len']/subject_info['fs']*self.std_freq)//self.sample_length)<=0 or (Y is None):
            return False,subject_info

        # print('??')
        return True,subject_info
    
    def get_xy_ecg_signals(self, subject_data, dataset_name):
        ecg_sig_name_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V','MLII']
        ecg_sig_index = [i  for i,name in enumerate(subject_data.sig_name) if name in ecg_sig_name_list]
        ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'ECG' in name.upper()]
        ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'II' in name.upper()]
        if dataset_name in ['NIFEADB','ADFECGDB','NIFECGDB','BA-LABOUR2']:
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'ABDOMEN' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'DIRECT' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'THORAX' in name.upper()]
        elif dataset_name in ['SensSmartTech','SensSmartTech_pcg'] :
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'CAROTID' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'BRACHIAL' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'PCG' in name.upper()]
        elif dataset_name in ['BIDMC','DALIA','WESAD'] :
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'PLETH' in name.upper()]
            ecg_sig_index = ecg_sig_index + [i  for i,name in enumerate(subject_data.sig_name) if 'PPG' in name.upper()]
            
            
        ecg_sig_index = self.dectect_abnormal(subject_data,ecg_sig_index)
        ecg_sig_name = np.array(subject_data.sig_name)[ecg_sig_index]
        ecg_sig_index = np.array(ecg_sig_index)
        # print(subject_data.sig_name)

        x_ecg_sig_name = None
        x_ecg_sig_index = None
        y_ecg_sig_name = None
        y_ecg_sig_index = None

        if dataset_name == 'ADFECGDB':
            x_ecg_sig_name = ecg_sig_name[~(ecg_sig_name=='Direct_1')].tolist()
            x_ecg_sig_index = ecg_sig_index[~(ecg_sig_name=='Direct_1')].tolist()
            y_ecg_sig_name = ecg_sig_name[(ecg_sig_name=='Direct_1')].tolist()
            y_ecg_sig_index = ecg_sig_index[(ecg_sig_name=='Direct_1')].tolist()
        elif dataset_name == 'BA-LABOUR2':
            x_ecg_sig_name = ecg_sig_name[~(ecg_sig_name=='fecg')].tolist()
            x_ecg_sig_index = ecg_sig_index[~(ecg_sig_name=='fecg')].tolist()
            y_ecg_sig_name = ecg_sig_name[(ecg_sig_name=='fecg')].tolist()
            y_ecg_sig_index = ecg_sig_index[(ecg_sig_name=='fecg')].tolist()
        elif dataset_name == 'ningbo':
            x_ecg_sig_name = ecg_sig_name[~(ecg_sig_name=='I')].tolist()
            x_ecg_sig_index = ecg_sig_index[~(ecg_sig_name=='I')].tolist()
            y_ecg_sig_name = ecg_sig_name[(ecg_sig_name=='I')].tolist()
            y_ecg_sig_index = ecg_sig_index[(ecg_sig_name=='I')].tolist()
        elif dataset_name == 'cpsc_2018':
            x_ecg_sig_name = ecg_sig_name[~(ecg_sig_name=='II')].tolist()
            x_ecg_sig_index = ecg_sig_index[~(ecg_sig_name=='II')].tolist()
            y_ecg_sig_name = ecg_sig_name[(ecg_sig_name=='II')].tolist()
            y_ecg_sig_index = ecg_sig_index[(ecg_sig_name=='II')].tolist()
        elif dataset_name == 'SensSmartTech':
            x_condition = list(map(lambda x:(("CAROTID" in x.upper()) | ("BRACHIAL" in x.upper())),ecg_sig_name))
            y_condition = list(map(lambda x:"II" in x,ecg_sig_name))
            x_ecg_sig_name = ecg_sig_name[x_condition].tolist()
            x_ecg_sig_index = ecg_sig_index[x_condition].tolist()
            y_ecg_sig_name = ecg_sig_name[y_condition].tolist()
            y_ecg_sig_index = ecg_sig_index[y_condition].tolist()
        elif dataset_name == 'SensSmartTech_pcg':
            x_condition = list(map(lambda x:"pcg" in x,ecg_sig_name))
            y_condition = list(map(lambda x:"II" in x,ecg_sig_name))
            x_ecg_sig_name = ecg_sig_name[x_condition].tolist()
            x_ecg_sig_index = ecg_sig_index[x_condition].tolist()
            y_ecg_sig_name = ecg_sig_name[y_condition].tolist()
            y_ecg_sig_index = ecg_sig_index[y_condition].tolist()
        elif dataset_name == 'BIDMC':
            x_condition = list(map(lambda x:"PLETH" in x,ecg_sig_name))
            y_condition = list(map(lambda x:"II" in x,ecg_sig_name))
            x_ecg_sig_name = ecg_sig_name[x_condition].tolist()
            x_ecg_sig_index = ecg_sig_index[x_condition].tolist()
            y_ecg_sig_name = ecg_sig_name[y_condition].tolist()
            y_ecg_sig_index = ecg_sig_index[y_condition].tolist()
        elif dataset_name in ['DALIA','WESAD']:
            x_condition = list(map(lambda x:"PPG" in x,ecg_sig_name))
            y_condition = list(map(lambda x:"ECG" in x,ecg_sig_name))
            x_ecg_sig_name = ecg_sig_name[x_condition].tolist()
            x_ecg_sig_index = ecg_sig_index[x_condition].tolist()
            y_ecg_sig_name = ecg_sig_name[y_condition].tolist()
            y_ecg_sig_index = ecg_sig_index[y_condition].tolist()
        elif dataset_name == 'ptb-xl':
            x_condition = list(map(lambda x:"II" == x,ecg_sig_name))
            y_condition = list(map(lambda x:"II" == x,ecg_sig_name))
            x_ecg_sig_name = ecg_sig_name[x_condition].tolist()
            x_ecg_sig_index = ecg_sig_index[x_condition].tolist()
            y_ecg_sig_name = ecg_sig_name[y_condition].tolist()
            y_ecg_sig_index = ecg_sig_index[y_condition].tolist()
        elif dataset_name == 'MITDB':
            x_condition = list(map(lambda x:"MLII" == x,ecg_sig_name))
            y_condition = list(map(lambda x:"MLII" == x,ecg_sig_name))
            x_ecg_sig_name = ecg_sig_name[x_condition].tolist()
            x_ecg_sig_index = ecg_sig_index[x_condition].tolist()
            y_ecg_sig_name = ecg_sig_name[y_condition].tolist()
            y_ecg_sig_index = ecg_sig_index[y_condition].tolist()

        # print(x_ecg_sig_name,y_ecg_sig_name)
        return x_ecg_sig_index,x_ecg_sig_name,y_ecg_sig_index,y_ecg_sig_name

    def get_ecg_signals(self, subject_data, dataset_name):
        ecg_sig_index,ecg_sig_name,_,_ = self.get_xy_ecg_signals(subject_data, dataset_name)
        return ecg_sig_index,ecg_sig_name

    def get_sample_list(self,):
        self.subject_list.loc[:,'n_sample'] = (self.subject_list['sig_len']/self.subject_list['fs']*self.std_freq)//self.sample_length
        self.subject_list = self.subject_list.loc[~(self.subject_list['ecg_n_sig'] <=0 |self.subject_list['n_sample'].isna() | (self.subject_list['n_sample'] <= 0) | (self.subject_list['n_sample'] == np.inf)),:]
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
            self._i_data.subject_path = subject_path
            if dataset_name in ['ptb-xl','MITDB']:
                X = []
                for snr in [-6, 0, 6, 12, 18, 24]:
                    X.append(np.expand_dims(self.add_noise(resample_signal[:,0],snr),axis=1))
                resample_signal = np.concat(X, axis=1)
            self._i_data.resample_signal = resample_signal
            self._i_data.dataset_name = self.sample_list.loc[index,'dataset_name']
            self._i_data.signal_mean = signal_mean
            self._i_data.signal_std = signal_std
            self._i_data.ecg_sig_name = ecg_sig_name
            self._i_data.Y = self.get_Y(self._i_data,self.sample_list.loc[index,'dataset_name'],os.path.join(self.dataset_path,subject_path))
        
        X = self._i_data.resample_signal[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1)),int(sample_sig)]
        Y = self._i_data.Y[int(self.sample_length*sample_index):int(self.sample_length*(sample_index+1))]
        return X,Y

    def get_Y(self,data,dataset_name,subject_path):
        Y = None
        _,_,y_ecg_sig_index,y_ecg_sig_name = self.get_xy_ecg_signals(data,dataset_name)
        if dataset_name in ['ADFECGDB','BA-LABOUR2','ningbo','cpsc_2018','SensSmartTech','SensSmartTech_pcg','BIDMC','DALIA','WESAD','MITDB','ptb-xl']:
            Y,_,_,_ = self.process_data(data,dataset_name,ecg_sig_index=y_ecg_sig_index,ecg_sig_name=y_ecg_sig_name)
            
        Y = Y.squeeze()
        if len(Y.shape) == 2:
            return None
        return Y

    def init_noise(self, root_path='../Data/GenerationDB/NSTDB/data'):
        """
        Initialize noise data by loading from the given path and resampling to the standard frequency.
        """
        noise_types = ['bw', 'ma', 'em']
        for noise_type in noise_types:
            # Read noise record
            noise_record = wfdb.rdrecord(os.path.join(root_path, noise_type))
            noise_data = noise_record.p_signal  # Get physical signal data
            
            # Handle multichannel data (use the first channel)
            if noise_data.ndim > 1:
                noise_data = noise_data[:, 0]
            
            # Compute target sample count
            original_samples = noise_data.shape[0]
            original_fs = noise_record.fs  # Original sampling frequency
            duration = original_samples / original_fs  # Noise duration (seconds)
            target_samples = int(duration * self.std_freq)  # Target sample count
            
            # Resample to the standard frequency
            resampled_noise = resample(noise_data, target_samples)
            
            # Store in dictionary (key=noise type, value=1D array)
            self.noise_dict[noise_type] = resampled_noise.flatten()

    def add_noise(self, signal, snr=0, noise_type='em'):
        """
        Add noise of a given type and SNR to the signal.
        - signal: input signal (1D array)
        - snr: signal-to-noise ratio in dB
        - noise_type: noise category ['bw', 'ma', 'em']
        """
        if len(self.noise_dict) == 0:
            self.init_noise()
        
        # Retrieve noise data
        if noise_type not in self.noise_dict:
            raise ValueError(f"Invalid noise type: {noise_type}. Valid options: {list(self.noise_dict.keys())}")
        noise = self.noise_dict[noise_type]
        
        # If the noise is shorter than the signal, repeat or truncate to match length
        if len(noise) < len(signal):
            # Tile noise to cover the full signal length
            noise_segment = np.tile(noise, (len(signal) // len(noise)) + 1)[:len(signal)]
        else:
            # Randomly select a noise segment matching signal length
            start_idx = np.random.randint(0, len(noise) - len(signal) + 1)
            noise_segment = noise[start_idx:start_idx + len(signal)]

        # Randomly select a noise segment matching signal length
        start_idx = np.random.randint(0, len(noise) - len(signal) + 1)
        noise_segment = noise[start_idx:start_idx + len(signal)]
        
        # Remove mean to eliminate DC component
        noise_segment = noise_segment - np.mean(noise_segment)
        
        # Compute signal and noise power
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise_segment ** 2)
        
        # Handle zero noise power edge case
        if noise_power == 0:
            scale = 1.0  # If noise power is zero, keep the original signal
        else:
            # Derive target noise power from the desired SNR
            target_noise_power = signal_power / (10 ** (snr / 10))
            # Compute scaling factor
            scale = np.sqrt(target_noise_power / noise_power)
        
        # Scale noise and add to the signal
        noisy_signal = signal + noise_segment * scale
        return noisy_signal
