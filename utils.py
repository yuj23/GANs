import torch
import random

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





