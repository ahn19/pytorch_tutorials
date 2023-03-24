
# # Transforms
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


from torch.utils.data import DataLoader
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(ds, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)


# ## ToTensor()
# ## Lambda Transforms
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))































