import os

import numpy as np
import wandb
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

class BaseValidation():
    def __init__(self,args:dict,model,train_data,test_data,is_finetune=True):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.args = args
        self.is_finetune = is_finetune
        self.device = torch.device(args['device'])

        if train_data is not None:
            self.train_dataloader = DataLoader(train_data,batch_size=self.args['batch_size'],shuffle=True)
            self.test_dataloader = DataLoader(test_data,batch_size=self.args['batch_size'],shuffle=False)
            self.val_dataloader = DataLoader(test_data,batch_size=self.args['batch_size'],shuffle=False)

        self.run_id = None
        self.run_name = self.args['run_name']
        if self.args['use_wandb']:
            self.run_id = self.init_wandb()

        self.model_save_path = os.path.join(self.args['model_root_path'],self.run_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.result = None

    def init_wandb(self):
        wandb.finish()
        api = wandb.Api()
        try:
            runs = api.runs(f"{self.args['username']}/{self.args['project_name']}")
        except :
            wandb.init(project=self.args['project_name'])
            wandb.finish()

        runs = api.runs(f"{self.args['username']}/{self.args['project_name']}",filters={"display_name":self.run_name})
        run_id = None

        for run in runs:
            if run.name == self.run_name:
                if self.args['is_load_pretrain_model']:
                    try:
                        run.delete()
                    except:
                        print(f'{self.run_name} Delete Failed!!!!')

        if self.run_id is None:
            wandb.init(project=self.args['project_name'], name=self.run_name,config=self.args)
            self.run_id = wandb.run.id

        return run_id

    def get_criterion(self):
        if self.args['criterion'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif self.args['criterion'] == 'MSELoss':
            return nn.MSELoss()
        elif self.args['criterion'] == 'BCELoss':
            return nn.BCELoss()
        
        return None

    def train_metric(self,label,predict):
        return accuracy_score(label, predict.argmax(dim=-1))

    def finetune(self,):
        self.model = self.model.to(self.device).train()
        batch_size = self.args['batch_size']
        learning_rate = self.args['learning_rate'] * np.sqrt(batch_size)
        epochs = self.args['epochs']
        num_batches = len(self.train_data) // batch_size + 1
        if self.args['use_wandb']:
            wandb.config.update({'traindata_length':len(self.train_data),'num_batches':num_batches})

        criterion = self.get_criterion()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_batches, eta_min=learning_rate/10)
        best_loss = 1e10

        for epoch in range(epochs):
            for i,(batch_X,batch_Y) in enumerate(self.train_dataloader):
                out = self.model(batch_X.to(self.device))
                loss = criterion(out, batch_Y.to(self.device))
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (i % (num_batches//self.args['monitor_iter']+1) == 0) & (self.args['use_wandb']):
                    wandb.log({'epoch':epoch, 'step':num_batches*epoch+i, 'loss':loss.item(),'lr':optimizer.param_groups[0]['lr']})
            
            for i,(batch_X,batch_Y) in enumerate(self.test_dataloader):
                with torch.no_grad():
                    out = self.model(batch_X.to(self.device))
                break
            
            train_metric = self.train_metric(label=batch_Y,predict=out.detach().cpu())
            if (self.args['use_wandb']):
                wandb.log({'epoch':epoch, 'train_metric':train_metric})
            
        if loss.item()<best_loss:
            best_loss = loss.item()
            torch.save(self.model.state_dict(), os.path.join(self.model_save_path,'best_model.pth'))

    def validation(self):
        if self.is_finetune:
            self.model.load_state_dict(torch.load(os.path.join(self.model_save_path,'best_model.pth'),weights_only=True))
        self.model = self.model.to(self.device).eval()
        if self.args['use_wandb']:
            wandb.config.update({'testdata_length': len(self.test_data)})

        model_out = []
        label = []
        X = []
        for i,(batch_X,batch_Y) in enumerate(self.test_dataloader):
            with torch.no_grad():
                out = self.model(batch_X.to(self.device))
            out = out.detach().cpu()
            model_out.append(out)
            label.append(batch_Y)
            X.append(batch_X)
        
        self.result = {'out':torch.concat(model_out,dim=0),'label':torch.concat(label, dim=0),'X':torch.concat(X, dim=0)}

    def save_result(self):
        torch.save(self.result, os.path.join(self.model_save_path,'result.pth'))

    def sample_analysis(self, idx=1):
        pass

    def summary(self):
        pass