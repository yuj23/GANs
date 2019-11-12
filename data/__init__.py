from data.datasets import CustomDataset
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from torchvision import transforms

def load_data(params):
    """
    1.creat dataset
    2.load data in specific batch size

    return data iterator
    """
    if params.dataset_name == 'horse2zebra':
        dataset = CustomDataset(params)
    if params.dataset_name == 'mnist':
        if not os.path.exists(params.dataset_path):
            os.makedirs(params.dataset_path)
        dataset = datasets.MNIST(root=params.dataset_path,train=params.isTrain,transform = params.transform,download=True) 
    dataloader = DataLoader(dataset,batch_size=params.batch_size, shuffle=params.shuffle,drop_last=True)
    return dataloader


