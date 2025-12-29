from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from validation.validation import BaseValidation


class ValClassification(BaseValidation):
    def __init__(self, args, model, train_data, test_data,is_finetune=True):
        super().__init__(args, model, train_data, test_data,is_finetune=is_finetune)

    def confusion_matrix(self):
        cm = confusion_matrix(self.result['label'], self.result['predict'], labels=list(range(self.train_data.num_class)))
        self.cm = cm

        cm_df = pd.DataFrame(cm, index=list(range(self.train_data.num_class)), columns=list(range(self.train_data.num_class)))
        if self.args['use_wandb']:
            table = wandb.Table(dataframe=cm_df)
            wandb.log({"confusion_matrix_table": table,"custom_step":0})

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap='Blues', ax=ax)
        plt.tight_layout()
        plt.close(fig)
        if self.args['use_wandb']:
            wandb.log({"confusion_matrix_image": wandb.Image(fig),"custom_step":0})

        return fig
    
    def summary(self):
        self.result['predict'] = self.result['out'].argmax(dim=-1)
        acc = accuracy_score(self.result['label'], self.result['predict'])
        precision, recall, f1_score, support = precision_recall_fscore_support(self.result['label'], self.result['predict'], average=None)

        metrics = np.stack([precision, recall, f1_score, support], axis=1)
        classes = np.array(list(range(self.args['num_class']))).reshape(-1, 1)  # Convert to column vector
        metrics = np.hstack([classes, metrics])
        columns = ["Class", "Precision", "Recall", "F1 Score", "Support"]
        self.metrics = pd.DataFrame(metrics, columns=columns)
        
        self.acc = acc
        if self.args['use_wandb']:
            wandb.log({"metrics_table": wandb.Table(data=self.metrics),"custom_step":0})
            wandb.log({'accuracy':acc,"custom_step":0})
        
        print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}, Support: {support}")
        return acc,precision, recall, f1_score, support

