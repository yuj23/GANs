import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import random


class CustomDataset(Dataset):
    def __init__(self,params):
        self.transform = transforms.Compose(params.transform)
        self.serial = params.serial
        self.isTrain = params.isTrain
        self.path_root = params.dataset_path
        self.data_name = params.dataset_name
        if self.isTrain == True:
            path_A = os.path.join(self.path_root,self.data_name,'trainA')
            path_B = os.path.join(self.path_root,self.data_name,'trainB')
        else:
            path_A = os.path.join(self.path_root,self.data_name,'testA')
            path_B = os.path.join(self.path_root,self.data_name,'testB')
        self.files_A = sorted([os.path.join(path_A,f) for f in os.listdir(path_A) if os.path.isfile(os.path.join(path_A,f))])
        self.files_B = sorted([os.path.join(path_B,f) for f in os.listdir(path_B) if os.path.isfile(os.path.join(path_B,f))])

    def __getitem__(self,index):
        A_path = self.files_A[index%len(self.files_A)]
        if not self.serial:
            indexB = random.randint(0,len(self.files_B)-1)
        else:
            indexB = index%len(self.files_B)
        B_path = self.files_B[indexB]
        A = self.transform(Image.open(A_path).convert('RGB'))
        B = self.transform(Image.open(B_path).convert('RGB'))
        return {'A':A,'B':B}

    def __len__(self):
        return max(len(self.files_A),len(self.files_B))

    

