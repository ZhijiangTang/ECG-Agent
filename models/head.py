
import os
import numpy as np
import torch
from torch import nn

from wfdb import processing

from models.mlp import BaseMLP
from utils.utils import frozen_model,filter_signal


class BaseFinetune(nn.Module):
    def __init__(self,args:dict, base_model=None,num_class=None,is_frozen=False):
        super().__init__()

        self.args = args
        self.model = base_model
        self.num_class = num_class
        self.is_frozen = is_frozen
        self.device = torch.device(self.args['device'])

        if is_frozen:
            self.model = frozen_model(self.model)
        self.mlp = self.init_mlp()
        
    def load_finetune_model(self):
        weight_path = os.path.join(self.args['model_root_path'], self.args['run_name'], self.args['model_save_name'])
        sd = torch.load(weight_path, weights_only=False)
        self.load_state_dict(sd, strict=True)

    def init_mlp(self):
        if self.num_class is not None:
            if len(self.args['hid_mlp'])==2:
                hid_nodes = []
            else:
                hid_nodes = [int(node) for node in self.args['hid_mlp'][1:-1].split(',')]

        return BaseMLP([self.args['oa_mlp_d_model']]+hid_nodes+[self.num_class],task_name='Classification')      
        

    def forward(self, x_enc):
        feature = self.model.finetune(x_enc)
        if len(feature.shape) == 2:
            feature = feature.unsqueeze(1)
        feature = feature.mean(1)
        out = self.mlp(feature)

        return out
    
    def save_model(self, model_save_path, model_save_name):
        torch.save(self.state_dict(), os.path.join(model_save_path,model_save_name))


class ClassificationFinetune(BaseFinetune):
    def __init__(self,args, base_model=None,num_class=None,is_frozen=False, **kwargs):
        super().__init__(args=args, base_model=base_model,num_class=num_class,is_frozen=is_frozen)


class DetectionFinetune(BaseFinetune):
    def __init__(self,args, base_model=None,is_frozen=False, **kwargs):
        super().__init__(args=args, base_model=base_model,is_frozen=is_frozen)

    def init_mlp(self):
        if len(self.args['hid_mlp'])==2:
            hid_nodes = []
        else:
            hid_nodes = [int(node) for node in self.args['hid_mlp'][1:-1].split(',')]

        return BaseMLP([self.args['oo_mlp_d_model']]+hid_nodes+[self.args['patch_len']],task_name='BioClassification')
    
    def forward(self, x_enc):
        batch_size,seq_len = x_enc.shape
        feature = self.model.finetune(x_enc)
        # print(feature.shape)
        if len(feature.shape) == 2:
            feature = feature.reshape(batch_size, seq_len, -1)
        
        out = self.mlp(feature)
        B,_,_ = out.shape
        out = out.reshape(B,-1)

        return out

class DetectionGQRS(DetectionFinetune):
    def __init__(self, args):
        self.model = None
        
        super().__init__(base_model=None,args=args)
        self.args = args

    def load_pretrain_model(self):
        pass

    def forward(self, x_enc,std_freq=100):
        device = x_enc.device
        B,T = x_enc.shape
        out = []
        for i in range(B):
            postions = processing.xqrs_detect(x_enc[i,:].detach().cpu().numpy(),fs=std_freq)
            Y = np.zeros([1,T])
            if len(postions)!=0:
                Y[:,postions] = 1
            
            out.append(Y)
        out = torch.tensor(np.concat(out,axis=0)).to(device)

        return out


class ForecastFinetune(BaseFinetune):
    def __init__(self,args, base_model=None,is_frozen=False, **kwargs):
        super().__init__(args=args, base_model=base_model,is_frozen=is_frozen)
        self.all_step = self.args['predict_length']//self.args['patch_len']

    def init_mlp(self):
        if len(self.args['hid_mlp'])==2:
            hid_nodes = []
        else:
            hid_nodes = [int(node) for node in self.args['hid_mlp'][1:-1].split(',')]

        return BaseMLP([self.args['oa_mlp_d_model']]+hid_nodes+[self.args['predict_length']])
    

    def forward(self, x_enc):
        _,T = x_enc.shape
        
        if 'Timer' in self.args['model_name']:
            out = self.Timer_forward(x_enc,self.all_step,0)
            return out[:,T:]
        elif 'UniTS' in self.args['model_name']:
            out = self.UniTS_forward(x_enc,self.all_step,0)
            return out[:,T:]
        elif 'Transformer' in self.args['model_name']:
            out = self.model.forecast(x_enc,self.all_step)
            return out
        else:
            feature = self.model.finetune(x_enc)
            if len(feature.shape) == 2:
                feature = feature.unsqueeze(1)
            feature = feature.mean(1)
            return self.mlp(feature).squeeze()
        
    def PSSM_forward(self, x_enc, all_step, step):
        if step == all_step:
            return x_enc.squeeze()
        # B,T = x_enc.shape
        # print(all_step,step)
        out = self.model.forecast(x_enc,all_step).squeeze(-1)
        predict = torch.concat([x_enc[:,self.args['patch_len']:], out],dim=1)
        return self.PSSM_forward(predict,all_step,step+1)

    def Timer_forward(self, x_enc, all_step, step):
        if step == all_step:
            return x_enc.squeeze()
        out = self.model.pretrain(x_enc).squeeze(-1)
        predict = torch.concat([x_enc,out[:,-self.args['patch_len']:]],dim=1)
        return self.Timer_forward(predict,all_step,step+1)


    def UniTS_forward(self, x_enc,all_step, step):
        if step == all_step:
            return x_enc.squeeze()
        out = self.model.forecast(x_enc, 1)
        # print(out.shape,x_enc.shape)
        predict = torch.concat([x_enc,out[:,-self.args['patch_len']:]],dim=1)
        return self.UniTS_forward(predict,all_step,step+1)


class GenerationFinetune(BaseFinetune):
    def __init__(self,args, base_model=None,is_frozen=False, **kwargs):
        super().__init__(args=args, base_model=base_model,is_frozen=is_frozen)

    def init_mlp(self):
        if len(self.args['hid_mlp'])==2:
            hid_nodes = []
        else:
            hid_nodes = [int(node) for node in self.args['hid_mlp'][1:-1].split(',')]

        return BaseMLP([self.args['oo_mlp_d_model']]+hid_nodes+[self.args['patch_len']])
    
    def forward(self, x_enc):
        batch_size,seq_len = x_enc.shape
        feature = self.model.finetune(x_enc)
        if len(feature.shape) == 2:
            feature = feature.reshape(batch_size, seq_len, -1)
        
        out = self.mlp(feature)
        B,_,_ = out.shape
        out = out.reshape(B,-1)

        return out


