import torch
import random
import os 
import matplotlib.pyplot as plt
import numpy as np

def make_sprite(data,inds,cols=2):
    """given a data [N,C,H,W],return sprire picture with [H,W*cols,C]"""
    pics = []
    for i in inds:
        pic = torch.unsqueeze(data[i],0)
        pics.append(pic)
    pics = torch.cat(pics,-1).squeeze(0)
    pics = pics.permute(1,2,0)
    return pics

def get_samples(model,data,epoch,params):
    """random sampling 2 pictures generated from generator and save it to save_path"""
    with torch.no_grad():
        fake_B = model.netG_A2B(data['A'].to(params.device))
    #save real image
    inds = np.random.choice(range(len(data)),2,replace=False) #2.Float()
    real_A = make_sprite(data['A'],inds)
    real_A = 127.5*(real_A.cpu().float().numpy()+1.0)
    real_A = real_A.astype(np.uint8)
    #form tensor to image with range [0-255]
    fake_B = make_sprite(fake_B,inds)
    fake_B = 127.5*(fake_B.cpu().float().numpy()+1.0)
    fake_B = fake_B.astype(np.uint8)
    if not os.path.exists(params.save_path):
        os.mkdir(params.save_path)
    name_fake = params.save_path+'/'+'epoch'+str(epoch)+'_fake_B_samples.jpg'
    name_real = params.save_path+'/'+'epoch'+str(epoch)+'_real_A_samples.jpg'
    plt.imsave(name_fake,fake_B)
    plt.imsave(name_real,real_A)
    print('Image saved.')
    return real_A,fake_B
 
def get_scheduler(optimizer,params):
    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - params.epoch_start_decay) / float(params.n_epoch-params.epoch_start_decay+1)
            return lr_l
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

class set_buffer_pool:
   """creat a buffer pool to choose images randomly with probability 0.5.
   if p > 0.5,choose the image previously stored in the pool,
   otherwise,return the input image.""" 
     
   def __init__(self,pool_size):
       self.pool_size = pool_size
    
   def choose(self,images):
       if self.pool_size == 0 or self.pool_size > len(images):
           return images
       data =[]
       return_images = []
       num_image = 0
       for i in images:
           i = torch.unsqueeze(i,0)
           if num_image < self.pool_size:
               return_images.append(i)
               data.append(i)
               num_image+=1
           else:
               p = random.uniform(0,1)
               if p > 0.5:#choose random image from pool
                   index = random.randint(0,self.pool_size-1)
                   tmp = data[index].clone()
                   return_images.append(tmp)
                   data[index]=i
               else:
                   return_images.append(i)
       return_images = torch.cat(return_images,0)
       return return_images    

def init_weight_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)



