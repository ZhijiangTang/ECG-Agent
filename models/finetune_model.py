
import os
import torch
from torch import nn
import torch.nn.functional as F

from models.FFD import FFD


class BaseLTM(nn.Module):
    def __init__(self,args:dict):
        super().__init__()

        self.args = args
        self.model = None
        self.device = torch.device(self.args['device'])

        self.load_pretrain_model()
        
        if not self.args['is_load_pretrain_model']:
            self.load_finetune_model()
        
    def load_pretrain_model(self):
        if 'Timer' in self.args['model_name']:
            import sys
            sys.path.append("..") 
            sys.path.append("../Timer")
            from Timer.models.model import Backbone
            sd = torch.load(os.path.join(self.args['model_root_path'], self.args['model_save_name']), map_location=self.device,weights_only=False)

            patch_len=self.args['patch_len']
            self.model = Backbone(patch_len=patch_len)
            self.model.load_state_dict(sd, strict=True)

        elif 'ECGPT' in self.args['model_name']:
            import sys
            sys.path.append("..") 
            sys.path.append("../HeartGPT") 
            from HeartGPT.model.model import HeartGPTModel
            sd = torch.load(os.path.join(self.args['model_root_path'], self.args['model_save_name']), map_location=self.device,weights_only=False)

            self.model = HeartGPTModel(vocab_size=101,device=self.device)
            self.model.load_state_dict(sd)

        elif 'UniTS' in self.args['model_name']:
            import sys
            sys.path.append("..") 
            sys.path.append('../UniTS')
            from UniTS.models.UniTS import UniTS
            from UniTS.exp.exp_pretrain import read_task_data_config,get_task_data_config_list

            if "Raw" in self.args['model_name']:
                sd = torch.load(os.path.join(self.args['model_root_path'], self.args['model_save_name']), map_location=self.device,weights_only=False)['student']
                # sd = {k[7:]: v for k, v in sd.items()}
            else:
                sd = torch.load(os.path.join(self.args['model_root_path'], self.args['model_save_name']), map_location=self.device,weights_only=False)
            patch_len=self.args['patch_len']
            task_data_config = read_task_data_config('../UniTS/data_provider/ecg_pretrain.yaml')
            task_data_config_list = get_task_data_config_list(task_data_config, default_batch_size=self.args['batch_size'])
            self.model = UniTS(configs_list=task_data_config_list,patch_len=patch_len,stride=patch_len)
            self.model.load_state_dict(sd,strict=False)
        
        elif 'ECG_PSSM' in self.args['model_name']:
            from _models.PSSM import PSSM

            sd = torch.load(os.path.join(self.args['model_root_path'], self.args['model_save_name']), map_location=self.device,weights_only=False)
            d_model = self.args['d_model']
            patch_len=self.args['patch_len']
            self.model = PSSM(d_model=d_model,mask_type='forward',predict_len=patch_len,is_pretrain=True)
            self.model.load_state_dict(sd,strict=False)
            
        elif 'Transformer' in self.args['model_name']:
            sd = torch.load(os.path.join(self.args['model_root_path'], self.args['model_save_name']), map_location=self.device,weights_only=False)
            self.model = FIDModel()
            self.model.load_state_dict(sd,strict=False)
    
    def load_finetune_model(self):
        weight_path = os.path.join(self.args['model_root_path'], self.args['run_name'], self.args['model_save_name'])
        sd = torch.load(weight_path, weights_only=False)
        self.load_state_dict(sd, strict=True)

    def pretrain(self, x_enc):
        return self.model.pretrain(x_enc)
    def forecast(self, x_enc,all_step):
        return self.model.forecast(x_enc,all_step)

    def finetune(self,x_enc):
        return self.model.finetune(x_enc)
    def forward(self, x_enc):
        feature = self.model.finetune(x_enc)

        return feature
    
    def save_model(self, model_save_path, model_save_name):
        torch.save(self.state_dict(), os.path.join(model_save_path,model_save_name))


class FlatenLine(nn.Module):
    def __init__(self,args=None, ):
        super(FlatenLine, self).__init__()
        self.args = args
        
    def finetune(self, x):
        return self.forward(x)
    def forward(self, x):
        
        
        return out

