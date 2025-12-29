from validation.Forecast import ValForecast
import numpy as np
import matplotlib.pyplot as plt
import wandb


class ValGeneration(ValForecast):
    def __init__(self, args, model, train_data, test_data,is_finetune=True):
        super().__init__(args, model, train_data, test_data,is_finetune=is_finetune)

    def sample_analysis(self,index):
        x = self.result['X'][index,...].numpy()
        predict = self.result['out'][index,...].numpy()
        y = self.result['label'][index,...].numpy()
        times = np.arange(len(y))

        fig, axes = plt.subplots(2, 1)

        axes[0].plot(times,y, label='True', color='b',alpha=0.8)
        axes[0].plot(times,predict, label='Predict', color='r',alpha=0.8)
        axes[0].legend(loc='upper left')  
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('ECG')

        axes[1].plot(times,x, label='X', color='b',alpha=1)
        axes[1].legend(loc='upper left')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('ECG')
        plt.tight_layout()
        fig.tight_layout()  
        plt.title('ECG Signal')

        if self.args['use_wandb']:
            wandb.log({"Show_Sample": wandb.Image(fig),'sample_index':index})
        
        plt.show()

