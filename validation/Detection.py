from sklearn.metrics import ConfusionMatrixDisplay,roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn.functional as F
from validation.validation import BaseValidation


class ValDetection(BaseValidation):
    def __init__(self, args, model, train_data, test_data,is_finetune=True):
        super().__init__(args, model, train_data, test_data,is_finetune=is_finetune)

    def roc(self):
        # Compute FPR, TPR and thresholds
        label = self.result['label'].flatten()
        out = self.result['out'].flatten()
        fpr, tpr, thresholds = roc_curve(label, out, pos_label=1)

        # AUC
        auc = roc_auc_score(label, out)
        print(f"AUC: {auc:.4f}")

        # ROC
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], color='r', linestyle='--') 
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Plot')
        ax.legend(loc='lower right')

        if self.args['use_wandb']:
            wandb.log({
            "roc_curve": wandb.Image(fig),
            "AUC": auc,
            "custom_step": 0
        })

        return fig
    
    def train_metric(self,label,predict):
        _,f1,_,_,_ = self.summary(label,predict)
        return f1

    def get_predict(self,label=None,out=None):
        if label is None:
            label = self.result['label'].flatten()
            out = self.result['out'].flatten()
        else:
            label = label.flatten()
            out_flatten = out.flatten()
        # Youden's J index（TPR - FPR）
        fpr, tpr, thresholds = roc_curve(label, out_flatten, pos_label=1)
        youden_j = tpr - fpr
        best_threshold_index = np.argmax(youden_j)
        best_threshold = thresholds[best_threshold_index]
        self.best_threshold = best_threshold

        threshold_p = out.clone()
        threshold_p[threshold_p <= best_threshold]=0
        predict = torch.cat([self.non_max_suppression(threshold_p[i])[0].unsqueeze(dim=0) for i in range(threshold_p.shape[0])],dim=0)
        return predict
    
    def get_tolerance_predict(self,label=None,predict=None,tolerance_time=70,std_freq=100):
        if label is None:
            label = self.result['label']
            predict = self.result['predict']
        # Within tolerance time window, treat predictions as correct
        tolerance_point = int(tolerance_time/1000*std_freq)//2*2+1

        # Use convolution to expand points around predict (after NMS, adjacent points should not appear)
        # todo: remove adjacent points if any
        expanded = F.conv1d(predict.unsqueeze(dim=1).float(), torch.ones(1, 1, tolerance_point), padding=tolerance_point//2)
        expanded = (expanded > 0).int().squeeze()
        # If expanded predict region contains label, mark as correct
        tolerance_predict_true = ((expanded) & (label.int()))
        # Use convolution to expand points around label
        expanded = F.conv1d(label.unsqueeze(dim=1).float(), torch.ones(1, 1, tolerance_point), padding=tolerance_point//2)
        expanded = (expanded > 0).int().squeeze()
        # In predict, set points around label to 0; set correct predictions to 1 for later metrics
        tolerance_predict = predict.clone()
        tolerance_predict[expanded.bool()] = 0
        tolerance_predict[tolerance_predict_true.bool()] = 1

        return tolerance_predict
    
    def summary(self,label=None ,out=None,tolerance_time=70,std_freq=100):
        save_result = False
        if label is None:
            save_result = True
            label = self.result['label']
            out = self.result['out']
        if self.args['model_name']=='GQRS':
            self.best_threshold = 1
            predict = out.clone()
        else:
            predict = self.get_predict(label=label,out=out)
        if save_result:
            self.result['predict'] = predict

        tolerance_predict = self.get_tolerance_predict(label=label,predict=predict,tolerance_time=tolerance_time,std_freq=std_freq)
        if save_result:
            self.result['tolerance_predict'] = tolerance_predict

        label = label.flatten()
        y_pred_best = tolerance_predict.flatten()
        # Compute confusion matrix
        cm = confusion_matrix(label, y_pred_best)
        acc = accuracy_score(label, y_pred_best)
        recall = recall_score(label, y_pred_best)
        f1 = f1_score(label, y_pred_best)

        self.acc = acc
        self.f1 = f1
        self.recall = recall
        self.cm = cm
        if self.args['use_wandb'] and save_result:
            wandb.log({'accuracy':acc,'f1':f1,'recall':recall,'best_threshold':self.best_threshold,"custom_step":0})

        print(f"Accuracy: {acc}, F1 Score: {f1}, Recall: {recall}, Best Threshold: {self.best_threshold}")
        return acc,f1,recall,self.best_threshold,cm
    
    def confusion_matrix(self):
        if self.cm is None:
            acc,f1,recall,best_threshold,cm = self.accuracy()
        cm = self.cm

        if self.args['use_wandb']:
            wandb.log({"confusion_matrix_table": wandb.Table(dataframe=pd.DataFrame(cm, index=list(range(2)), columns=list(range(2)))),"custom_step":0})

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap='Blues', ax=ax)
        plt.tight_layout()
        plt.close(fig)
        if self.args['use_wandb']:
            wandb.log({"confusion_matrix_image": wandb.Image(fig),"custom_step":0})

        return fig
    
    def sample_analysis(self,index):
        ecg_data = self.result['X'][index,...].numpy()
        detection_p = self.result['out'][index,...].numpy()
        detection_y = self.result['predict'][index,...].numpy()
        detection_true = self.result['label'][index,...].numpy()

        fig, ax1 = plt.subplots(figsize=(12, 4))

        ax1.plot(ecg_data, label='ECG Signal', color='b',alpha=0.2)
        valid_qrs_indices = np.where(detection_y == 1)[0]
        ax1.scatter(valid_qrs_indices, ecg_data[valid_qrs_indices], color='r', label='Predict', s=25,alpha=0.5)
        true_qrs_indices = np.where(detection_true==1)[0]
        ax1.scatter(true_qrs_indices, ecg_data[true_qrs_indices], color='g', label='True', s=50,alpha=0.5)

        ax1.set_xlabel('Time')
        ax1.set_ylabel('ECG Amplitude', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(detection_p, label='Detection Probability', color='r', alpha=0.3)
        ax2.plot(np.arange(len(detection_p)),np.full(len(detection_p),self.best_threshold), color='r')
        ax2.set_ylabel('Detection Probability', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.legend(loc='upper left') 
        ax2.legend(loc='upper right') 
        plt.legend()
        fig.tight_layout() 
        plt.title('ECG Signal with Detection Probability')

        if self.args['use_wandb']:
         wandb.log({"Show_Sample": wandb.Image(fig),'sample_index':index})
        
        plt.show()

    def non_max_suppression(self, scores, threshold=19):
        """
        Perform non-maximum suppression (NMS)
        """
        index = scores.argsort(descending=True)
        index = index[scores[index] != 0]

        result_index = []
        while len(index) > 0:
            result_index.append(index[0])
            index = index[(index-result_index[-1]).abs()>threshold]

        result_index = torch.tensor(result_index).int()
        result = torch.zeros_like(scores)
        result[result_index] = 1
        return result, result_index
