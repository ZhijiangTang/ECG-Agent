from validation.validation import BaseValidation
from models.FFD import FFDModel
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import math
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm


class ValForecast(BaseValidation):
    def __init__(self, args, model, train_data, test_data,is_finetune=True):
        super().__init__(args, model, train_data, test_data,is_finetune=is_finetune)
        self.ffd_feature_model = self.load_ffd_feature_model().to(self.device)

    def train_metric(self, label, predict):
        ffd,_ = self.FeatureDistance(label, predict)
        return ffd
    
    def FreqMSE(self):
        out = self.result['out']
        label = self.result['label']
        
        out_amp = torch.abs(torch.fft.fft(out))
        label_amp = torch.abs(torch.fft.fft(label))
        epsilon = 1e-8
        
        out_mean = out_amp.mean(dim=-1, keepdim=True)
        out_std = out_amp.std(dim=-1, keepdim=True)
        out_normalized = (out_amp - out_mean) / (out_std + epsilon)
        
        label_mean = label_amp.mean(dim=-1, keepdim=True)
        label_std = label_amp.std(dim=-1, keepdim=True)
        label_normalized = (label_amp - label_mean) / (label_std + epsilon)
        
        freq_mse = torch.mean((out_normalized - label_normalized) ** 2)
        
        if self.args['use_wandb']:
            wandb.log({'FreqMSE': freq_mse, 'custom_step': 0})
        return freq_mse

    def PSNR(self):
        max_value = torch.max(self.result['label'])
        psnr = 10 * torch.log10(max_value ** 2 / torch.square(self.result['label'] - self.result['out']).mean())
        if self.args['use_wandb']:
            wandb.log({'PSNR': psnr, 'custom_step': 0})
        return psnr

    def load_ffd_feature_model(self):
        
        sd = torch.load(os.path.join(self.args['ffd_model_root_path'], self.args['ffd_model_save_name']), map_location=self.device,weights_only=False)

        model = FFDModel()
        model.load_state_dict(sd, strict=False)

        return model
    
    def get_ffd_feature(self,label=None,out=None):
        if label is None:
            label = self.result['label']
            out = self.result['out']

        dataloader = DataLoader(out, batch_size=int(self.args['ffd_batch_size']), shuffle=False)
        out_feature = []
        for i, x in enumerate(dataloader):
            with torch.no_grad():
                transformer_feature = self.ffd_feature_model.ffd(x.to(self.device))
                transformer_feature = transformer_feature.mean(dim=1)
            out_feature.append(transformer_feature.detach().cpu())
        out_feature = torch.cat(out_feature, dim=0)
        
        dataloader = DataLoader(label, batch_size=int(self.args['ffd_batch_size']), shuffle=False)
        label_feature = []
        for i, x in enumerate(dataloader):
            with torch.no_grad():
                transformer_feature = self.ffd_feature_model.ffd(x.to(self.device))
                transformer_feature = transformer_feature.mean(dim=1)
            label_feature.append(transformer_feature.detach().cpu())
        label_feature = torch.cat(label_feature, dim=0)
        return out_feature, label_feature
    
    def FeatureDistance(self,label=None,out=None):
        save_result = False
        if label is None:
            save_result = True
            label = self.result['label']
            out = self.result['out']

        out_feature,label_feature = self.get_ffd_feature(label=label,out=out)
        if save_result:
            self.result['out_feature'],self.result['label_feature'] = out_feature,label_feature
        ffd ,correct_ffd= self.FFD(label_feature=label_feature,out_feature=out_feature)

        if self.args['use_wandb'] and save_result:
            wandb.log({'FFD': ffd,'correct_ffd':correct_ffd, 'custom_step': 0})

        return ffd, correct_ffd

    def FFD(self, label_feature=None, out_feature=None):
        if label_feature is None:
            out_feature,label_feature  = self.result['out_feature'],self.result['label_feature']

        mu1, sigma1 = torch.mean(out_feature, dim=0), torch.cov(out_feature.T)
        mu2, sigma2 = torch.mean(label_feature, dim=0), torch.cov(label_feature.T)
        k = mu2.shape[0]
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        ffd = torch.sum((mu1 - mu2) ** 2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
        correct_ffd = ffd/math.sqrt(k)

        return ffd, correct_ffd

    def QSNR(self):
        signal_power = torch.square(self.result['label']).mean()
        noise_power = torch.square(self.result['label'] - self.result['out']).mean()
        qs_nr = 10 * torch.log10(signal_power / noise_power)
        if self.args['use_wandb']:
            wandb.log({'QSNR': qs_nr, 'custom_step': 0})
        return qs_nr

    def MSE(self):
        mse = (torch.square(self.result['out']-self.result['label']).mean())
        if self.args['use_wandb']:
            wandb.log({'MSE':mse,'custom_step':0})
        return mse

    def sample_analysis(self,index):
        x = self.result['X'][index,...].numpy()
        predict = self.result['out'][index,...].numpy()
        y = self.result['label'][index,...].numpy()
        times = np.arange(len(x)+len(y))

        fig, ax1 = plt.subplots(figsize=(12, 4))

        ax1.plot(times[:len(x)],x, label='Input ECG', color='b',alpha=1)
        ax1.plot(times[len(x):],y, label='True', color='b',alpha=0.5)
        ax1.plot(times[len(x):],predict, label='Predict', color='r',alpha=0.5)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('ECG')

        ax1.legend(loc='upper left')  
        plt.legend()
        fig.tight_layout()  
        plt.title('ECG Signal')
        if self.args['use_wandb']:
            wandb.log({"Show_Sample": wandb.Image(fig),'sample_index':index})
        
        plt.show()

    def summary(self):
        mse = self.MSE()
        psnr = self.PSNR()
        ffd = self.FeatureDistance()
        freq_mse = self.FreqMSE()
        qsnr = self.QSNR()

        print(f"MSE: {mse}, PSNR: {psnr}, FFD: {ffd}, FreqMSE: {freq_mse}, QSNR: {qsnr}")

        return mse, psnr, ffd, freq_mse, qsnr