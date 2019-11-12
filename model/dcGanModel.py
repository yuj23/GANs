from model.base_model import BaseModel 
import torch
import torch.nn as nn

class D(nn.Module):
    def __init__(self,in_channel):
        super(D,self).__init__()
        self.conv = nn.Sequential(
              #28-1
        nn.Conv2d(in_channel,512,3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2),
              #14-7
        nn.Conv2d(512,256,3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
              #7-
        nn.Conv2d(256,128,3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.AvgPool2d(4),)

        self.fc = nn.Sequential(
              nn.Linear(128,1),
              nn.Sigmoid(),)
    def forward(self,x,y=None):
        x = self.conv(x)
        x = self.fc(x.view(x.size(0),-1))
        return x


class G(nn.Module):
    """
    convolutiona generator for MNIST"""
    def __init__(self,input_size,out_channel):
        super(G,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,4*4*512),
            nn.ReLU(),)
        self.conv = nn.Sequential(
            #input 4 by 4 output 7 by 7
            nn.ConvTranspose2d(512,256,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #input 7 by 7 output 14 by 14
            nn.ConvTranspose2d(256,128,4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #input14 by 14,output 28 by 28
            nn.ConvTranspose2d(128,out_channel,4,stride=2,padding=1,bias=False),
            nn.Tanh(),)
    def forward(self,x,y=None):
        x = x.view(x.size(0),-1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0),512,4,4)
        y_ = self.conv(y_)
        return y_

class DCGAN(BaseModel):
    def __init__(self,params):
        super(DCGAN,self).__init__(params)
        self.loss_names = ['loss_G','loss_D','loss']
        self.losses = self.make_loss_dict()    
        self.model_names = ['G','D']
        self.netG = G(self.params.n_noise,self.params.output_nc).to(self.params.device)
        self.netD = D(self.params.input_nc).to(self.params.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=self.params.lr,betas=self.params.betas)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=self.params.lr,betas=self.params.betas)
        self.optimizers = [self.optimizer_G,self.optimizer_D]
        self.target_real = torch.ones(self.params.batch_size,requires_grad=False).to(self.params.device)
        self.target_fake = torch.zeros(self.params.batch_size,requires_grad=False).to(self.params.device)
        self.init_weight_normal()

    def set_input(self,data):
        self.data = data.to(self.params.device)
        self.noise = torch.randn(self.params.batch_size,self.params.n_noise).to(self.params.device)

    def G_step(self):
        x = self.netG(self.noise)
        self.loss_G = self.criterion(self.netD(x),self.target_real)
        self.requires_grad([self.netD],False)
        self.netG.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.requires_grad([self.netD],True)

    def D_step(self):
        self.loss_D_fake = self.criterion(self.netD(self.netG(self.noise)),self.target_fake)
        self.loss_D_real = self.criterion(self.netD(self.data),self.target_real)
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.netD.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

    def save_loss(self):
        self.losses['loss_D'].append(self.loss_D.item())
        self.losses['loss_G'].append(self.loss_G.item())
        self.losses['loss'].append(self.loss_D.item()+self.loss_G.item())

        
    


