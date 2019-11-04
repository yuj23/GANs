from data.datasets import CustomDataset
from torch.utils.data import DataLoader

def load_data(params):
    """
    1.creat dataset
    2.load data in specific batch size

    return data iterator
    """
    dataset = CustomDataset(params)
    dataloader = DataLoader(dataset,batch_size=params.batch_size, shuffle=params.shuffle)
    return dataloader


