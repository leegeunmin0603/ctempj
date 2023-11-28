import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, data_dir='./data', ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        

    def setup(self, stage=None , validation_split=0.2):        
        
        if stage is None:
            
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            self.dataset = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
            
            subset_size = 800
            self.dataset = torch.utils.data.Subset(self.dataset, range(subset_size))
            
            dataset_size = len(self.dataset)
            val_size = int(validation_split * dataset_size)
            train_size = dataset_size - val_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)