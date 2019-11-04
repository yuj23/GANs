"""
All base network model included such like generator_A2B, Discriminator_A.
"""
import torch.nn as nn
import torch.nn.functional as F

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


def get_scheduler(optimizer,params):
    def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - params.epoch_start_decay) / float(params.n_epoch-params.epoch_start_dacay+1)
            return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class G(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(G, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class D(nn.Module):
    def __init__(self, input_nc):
        super(D, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


