
"""
save all parameters.
"""
import torch 
from torchvision import transforms
from PIL import Image

class Params:
    def __init__(self,name=None):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if name == 'DCGAN':
            self.dcGanParams()
        if name == 'cycleGan':
            self.cycleGanParams()
        
    def dcGanParams(self):
        self.save_path = './save/dcGan'
        #load data
        self.isTrain = True 
        self.batch_size = 64
        self.shuffle = True
        self.dataset_name = 'mnist'
        self.dataset_path = './data/mnist/'
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])  
        #train parasm
        self.n_epoch=30
        self.n_critic = 5 #generator step size
        self.n_noise = 100
        self.epoch = 1 #starting epoch
        #hyper parameters
        self.lr = 0.001
        self.betas = (0.5,0.999)
        #print
        self.print_freq = 1024*10 
        self.save_latest_freq = 1024*10 
        self.save_epoch_freq = 5 
        #model 
        self.input_nc = 1
        self.output_nc = 1
        
    def cycleGanParams(self):
        self.save_path = "./save/cyclegan" 
        self.isTrain = True
        self.transform =[transforms.Resize(int(256*1.12), Image.BICUBIC), 
                transforms.RandomCrop(256), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        self.dataset_path = './data'
        self.dataset_name='horse2zebra'
        self.serial = True
        self.hy_params()
        self.print_params()
        self.model_params()
        self.data_params()

    def hy_params(self):
        self.n_epoch = 300  #number of epoch
        self.epoch = 1  # start epoch
        self.epoch_start_decay = 50 # start to decay lerning rate
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lambda_idt_A = 5
        self.lambda_idt_B = 5
        self.lambda_cycle_A = 10
        self.lambda_cycle_B = 10

    def print_params(self):
        self.print_freq = 100
        self.save_lasted_freq = 512
        self.save_epoch_freq = 15

    def model_params(self):
        self.pool_size = 2
        self.input_nc = 3
        self.output_nc = 3
        
    def data_params(self):
        self.num_threads= 4
        self.shuffle = True
        self.batch_size = 2




