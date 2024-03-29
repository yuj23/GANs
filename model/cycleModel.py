"""
The cycle gan model is subclass of BaseModel.
"""
from model.base_model import BaseModel
import torch
import itertools
from model.base_networks import G
from model.base_networks import D
from utils import set_buffer_pool
from utils import get_scheduler

class cycleGAN(BaseModel):
    """
    This is a implementation of the cycleGAN model,learning image-to-image transformation without paired data.
    paper link: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,params):
        BaseModel.__init__(self,params)
        self.loss_names = ['loss_identity','loss_cycle','loss_gan','loss_G','loss_D','loss_D_A','loss_D_B']
        self.losses = self.make_loss_dict()
        if self.params.isTrain:
            self.model_names = ['G_A2B','G_B2A','D_A','D_B']
        else:
            self.model_names = ['G_A2B','G_B2A']
        # define image pool
        self.fake_pool_A = set_buffer_pool(self.params.pool_size)
        self.fake_pool_B = set_buffer_pool(self.params.pool_size)
        # define the nets
        self.netG_A2B = G(self.params.input_nc,self.params.output_nc).to(self.device)
        self.netG_B2A = G(self.params.input_nc,self.params.output_nc).to(self.device)
        self.netD_A = D(self.params.input_nc).to(self.device)
        self.netD_B = D(self.params.input_nc).to(self.device)
        # define optimizer
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(),\
             self.netG_B2A.parameters()),lr=params.lr,betas=(params.beta1,0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),\
             self.netD_B.parameters()),lr=params.lr, betas=(params.beta1,0.999))
        self.optimizers = [self.optimizer_G,self.optimizer_D]
        if self.params.isTrain:
            self.schedulers=[get_scheduler(optim,self.params) for optim in self.optimizers]
        # define loss function
        self.criterion_identity = torch.nn.L1Loss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_GAN = torch.nn.MSELoss()
        #create target real and fake
        self.target_real = torch.ones(self.params.batch_size,requires_grad=False).to(self.params.device)
        self.target_fake = torch.zeros(self.params.batch_size,requires_grad=False).to(self.params.device)
        #init net 
        self.init_weight_normal()
        

    def set_input(self,data):
        """
        put set data into the 'device'.
        """
        self.real_A = data['A'].to(self.params.device)
        self.real_B = data['B'].to(self.params.device)


    def forward(self):
        """
        do forward to get ouput.
        """
        self.fake_A = self.netG_B2A(self.real_B)
        self.recover_A = self.netG_B2A(self.netG_A2B(self.real_A))
        self.fake_B = self.netG_A2B(self.real_A)
        self.recover_B = self.netG_A2B(self.netG_B2A(self.real_B))

    def backward_G(self):
        """
        claculate Generator loss and backward to get gradient.
        """
        # identity loss
        self.loss_idt_A = self.criterion_identity(self.netG_B2A(self.real_A),self.real_A)
        self.loss_idt_B = self.criterion_identity(self.netG_A2B(self.real_B),self.real_B)
        self.loss_idt = self.loss_idt_A * self.params.lambda_idt_A + self.loss_idt_B * self.params.lambda_idt_B
        # cycle loss
        self.loss_cycle_A = self.criterion_cycle(self.netG_A2B(self.fake_A),self.real_B)
        self.loss_cycle_B = self.criterion_cycle(self.netG_B2A(self.fake_B),self.real_A)
        self.loss_cycle = self.loss_cycle_A * self.params.lambda_cycle_A + self.loss_cycle_B * self.params.lambda_cycle_B
        # gan loss
        self.loss_gan_A_fake = self.criterion_GAN(self.netD_A(self.fake_A),self.target_real)
        self.loss_gan_B_fake = self.criterion_GAN(self.netD_B(self.fake_B),self.target_real)
        self.loss_gan = self.loss_gan_A_fake + self.loss_gan_B_fake
        self.loss_G = self.loss_idt + self.loss_cycle + self.loss_gan_A_fake + self.loss_gan_B_fake
        self.loss_G.backward()

    def save_loss(self):
        self.losses['loss_identity'].append(self.loss_idt.item())
        self.losses['loss_cycle'].append(self.loss_cycle.item())
        self.losses['loss_gan'].append(self.loss_gan.item())
        self.losses['loss_G'].append(self.loss_G.item())
        self.losses['loss_D_A'].append(self.loss_D_A.item())
        self.losses['loss_D_B'].append(self.loss_D_B.item())
        self.losses['loss_D'].append(self.loss_D.item())

    def backward_D(self):
        """
        calculate Discriminator loss and backward to get gradient.
        """
        self.loss_D_A_real = self.criterion_GAN(self.netD_A(self.real_A),self.target_real)*0.5
        fake_A = self.fake_pool_A.choose(self.fake_A)
        self.loss_D_A_fake = self.criterion_GAN(self.netD_A(fake_A.detach()),self.target_fake)*0.5
        self.loss_D_B_real = self.criterion_GAN(self.netD_B(self.real_B),self.target_real)*0.5 
        fake_B = self.fake_pool_B.choose(self.fake_B)
        self.loss_D_B_fake = self.criterion_GAN(self.netD_B(fake_B.detach()),self.target_fake)*0.5
        self.loss_D = self.loss_D_A_real + self.loss_D_A_fake + self.loss_D_B_real + self.loss_D_B_fake
        self.loss_D_A = self.loss_D_A_fake + self.loss_D_A_real
        self.loss_D_B = self.loss_D_B_fake + self.loss_D_B_real
        self.loss_D_A.backward()
        self.loss_D_B.backward()

    def step(self):
        """
        update the network paramters.
        """
        self.forward()
        #update Generator
        self.requires_grad([self.netD_A,self.netD_B],False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        #update Discriminator
        self.requires_grad([self.netD_A,self.netD_B],True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
    

