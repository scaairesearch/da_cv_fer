from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import os

class RunConfig:
    def __init__(self) -> None:
        self.EPOCHS = 2
        self.NUM_EPOCHS = 2
        self.IN_COLAB = False
        if os.path.exists('/content'): # check for google colab
            self.IN_COLAB = True
        
        if self.IN_COLAB:
            self.EPOCHS = 20
            self.NUM_EPOCHS = 20  
        # self.optimizer = None
        self.lr_strategy = None
        self.inital_lr = 0.005
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.criterion_class = nn.CrossEntropyLoss() # we are not doing log_softmax or softmax
        self.criterion_domain = nn.BCEWithLogitsLoss() # we can also use nn.CrossEntropyLoss() - we are not doing log_softmax or softmax

        self.scheduler_patience = 3
        self.scheduler_threshold = 0.001
        self.scheduler_factor =0.2

        self.early_stop_patience = 8
        self.early_stop_difference = 0.001
        self.EWC_LAMBDA = 0.4
        


if __name__ =='__main__':
    runconfig = RunConfig()
    print(runconfig.EPOCHS)
    print(runconfig.device)
    
