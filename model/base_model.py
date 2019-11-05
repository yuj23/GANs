"""
This is a basic model class for every GAN model,all GAN models should have these properties.
"""

import os
import torch
from utils import get_scheduler

class BaseModel:

    def __init__(self,params):
        self.params = params
        self.epoch = self.params.epoch
        self.loss_names = [] 
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
    def set_input(self,data):
        """feed the data"""
        pass

    def forward(self):
        """
        put data and forward to get some results
        """
        pass
    
    def backward_G(self):
        """
        Calculate loss,gradient and do backward for Generator.
        """
        pass

    def backward_D(self):
        """
        Calculate loss,gradient and do backward for Discriminator.
        """
        pass

    def step(self):
        """
        update the parameters and network.
        """
        pass

    def make_loss_dict(self):
        """get loss"""
        losses = {}
        for name in self.loss_names:
            losses[name] = [] 
        return losses

    def update_learning_rate(self):
        """update learning rate"""
        for scheduler in self.schedulers:
            scheduler.step()

    def load(self,name='latest'):
        """load latest or epoch checkpoint"""
        filename = self.params.save_path+'/'+name+'_checkpoint.pth.tar'
        if not os.path.exists(filename):
            print('Load model : Start a new train')
            return
        state = torch.load(filename,map_location=self.device)
        for name in state.keys():
            if ('net' or 'optimizer') in name:
                full_name = getattr(self,name)
                full_name.load_state_dict(state[name])
            else:
                full_name = getattr(self,name)
                full_name = state[name]
        print('Load model : Model loaded done!')
 
                   
    def save_model(self,epoch):
        """save epoch,model, optimizers and losses to resume training later"""
        dicts = {}
        dicts['epoch']=self.epoch
        dicts['losses'] = self.losses
        dicts['optimizer_G'] = self.optimizer_G
        dicts['optimizer_D'] = self.optimizer_D
        if not os.path.exists(self.params.save_path):
            os.mkdir(self.params.save_path)
        save_path = self.params.save_path+'/'+str(epoch)+'_checkpoint.pth.tar'
        for name in self.model_names:
            if isinstance(name,str):
                name = 'net'+name
                net = getattr(self,name)
                if torch.cuda.is_available():
                    dicts[name] = net.cpu().state_dict()
                    net.cuda()
                else:
                    dicts[name] = net.state_dict()
        torch.save(dicts,save_path)

