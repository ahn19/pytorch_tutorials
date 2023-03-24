#!/usr/bin/env python
# coding: utf-8

# 딥러닝 모델이 이미지를 학습하기 전 **이미지 정규화**를 진행하는 것은 **일반적으로 수행하는 전처리** 입니다.
# 
# 이미지 정규화를 진행하는 대표적인 이유 중 하나는 오차역전파(backpropagation)시, 그라디언트(Gradient) 계산을 수행하게 되는데, 데이터가 유사한 범위를 가지도록 하기 위함입니다.
# 
# 하지만, 정규화를 어떻게 수행하는가에 따라서 모델의 학습결과는 달라질 수 있습니다.
# 
# 이번에는 다양한 정규화 적용 방법에 대하여 알아보고, 정규화된 결과가 어떻게 달라지는지 확인해 보도록 하겠습니다.

# ## 모듈 import

# In[1]:


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ## `torchvision.transforms`

# `torchvision`의 `transforms`를 활용하여 정규화를 적용할 수 있습니다.

# `transoforms.ToTensor()` 외 다른 Normalize()를 적용하지 않은 경우
# 
# - 정규화(Normalize) 한 결과가 **0 ~ 1** 범위로 변환됩니다.

# In[2]:


transform = transforms.Compose([
    # 0~1의 범위를 가지도록 정규화
    transforms.ToTensor(),
])


# `datasets.CIFAR10` 데이터셋을 로드 하였습니다.
# 
# 데이터셋 로드시 `transform` 옵션에 지정할 수 있습니다.

# In[3]:


# datasets의 CIFAR10 데이터셋 로드 (train 데이터셋)
train = datasets.CIFAR10(root='data', 
                         train=True, 
                         download=True, 
                         # transform 지정
                         transform=transform                
                        )


# In[4]:


# datasets의 CIFAR10 데이터셋 로드 (test 데이터셋)
test = datasets.CIFAR10(root='data', 
                        train=False, 
                        download=True, 
                        # transform 지정
                        transform=transform
                       )


# In[23]:


# 이미지의 RGB 채널별 통계량 확인 함수
def print_stats(dataset):
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')
    
    min_r = np.min(imgs, axis=(2, 3))[:, 0].min()
    min_g = np.min(imgs, axis=(2, 3))[:, 1].min()
    min_b = np.min(imgs, axis=(2, 3))[:, 2].min()

    max_r = np.max(imgs, axis=(2, 3))[:, 0].max()
    max_g = np.max(imgs, axis=(2, 3))[:, 1].max()
    max_b = np.max(imgs, axis=(2, 3))[:, 2].max()

    mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
    mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
    mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

    std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
    std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
    std_b = np.std(imgs, axis=(2, 3))[:, 2].std()
    
    print(f'min: {min_r, min_g, min_b}')
    print(f'max: {max_r, max_g, max_b}')
    print(f'mean: {mean_r, mean_g, mean_b}')
    print(f'std: {std_r, std_g, std_b}')


# `transforms.ToTensor()`만 적용한 경우, 모든 이미지의 픽셀 값이 `0~1`의 범위를 가지도록 변환되었습니다.

# In[6]:


print_stats(train)
print('==='*10)
print_stats(test)


# ## `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` 적용

# 이번에는 `Normalize()` 함수 안에 `(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)`로 적용하겠습니다.
# 
# - 정규화(Normalize) 한 결과가 **-1 ~ 1** 범위로 변환됩니다.

# In[7]:


transform = transforms.Compose([
    transforms.ToTensor(),
    # -1 ~ 1 사이의 범위를 가지도록 정규화
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[8]:


# datasets의 CIFAR10 데이터셋 로드 (train 데이터셋)
train = datasets.CIFAR10(root='data', 
                         train=True, 
                         download=True, 
                         transform=transform                
                        )


# In[9]:


# datasets의 CIFAR10 데이터셋 로드 (test 데이터셋)
test = datasets.CIFAR10(root='data', 
                        train=False, 
                        download=True, 
                        transform=transform
                       )


# 아래 통계에서 확인할 수 있듯이, 이미지의 픽셀 값의 범위가 `0 ~ 1` 이 아닌 `-1 ~ 1` 사이의 범위를 가지도록 변환 되었습니다.

# In[10]:


print_stats(train)
print('==='*10)
print_stats(test)


# ## `transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),` 적용시
# 
# - ImageNet이 학습한 수백만장의 이미지의 RGB 각각의 채널에 대한 평균은 `0.485`, `0.456`, `0.406` 그리고 표준편차는 `0.229`, `0.224`, `0.225` 입니다. 만약, 일반적인 조도, 각도, 배경을 포함하는 평범한 이미지의 경우는 `(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)`으로 정규화하는 것을 추천한다는 커뮤니티 의견이 지배적입니다.
# 
# - 하지만, 전혀 새로운 이미지 데이터를 학습할 경우는 이 다음 섹션에서 가지고 있는 데이터셋에 대한 평균, 표준편차를 산출하여 적용할 수 있습니다.

# In[11]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# In[12]:


# datasets의 CIFAR10 데이터셋 로드 (train 데이터셋)
train = datasets.CIFAR10(root='data', 
                         train=True, 
                         download=True, 
                         transform=transform                
                        )


# In[13]:


# datasets의 CIFAR10 데이터셋 로드 (test 데이터셋)
test = datasets.CIFAR10(root='data', 
                        train=False, 
                        download=True, 
                        transform=transform
                       )


# In[14]:


print_stats(train)
print('==='*10)
print_stats(test)


# ## 데이터셋의 평균(mean)과 표준편차(std)를 계산하여 적용시
# 
# - 학습할 이미지 데이터셋이 일반적인 조도, 각도, 배경, 사물체가 아닌 경우는 직접 평균/표준편차를 계산하여 적용할 수 있습니다.

# 아래 함수는 이미지 데이터셋에 대하여 평균, 표준편차를 산출해 주는 함수 입니다.

# In[15]:


def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


# In[16]:


# 먼저, 변환하기 전 이미지 데이터셋을 로드 하기 위하여 transforms.ToTensor() 만 적용합니다.
transform = transforms.Compose([
    transforms.ToTensor(),
])


# In[17]:


# datasets의 CIFAR10 데이터셋 로드 (train 데이터셋)
train = datasets.CIFAR10(root='data', 
                         train=True, 
                         download=True, 
                         transform=transform                
                        )


# 계산된 평균과 표준편차는 다음과 같습니다.

# In[18]:


mean_, std_ = calculate_norm(train)
print(f'평균(R,G,B): {mean_}\n표준편차(R,G,B): {std_}')


# 이제 계산된 평균과 표준편차를 적용하여 변환합니다.

# In[19]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_, std_),
])


# In[20]:


# datasets의 CIFAR10 데이터셋 로드 (train 데이터셋)
train = datasets.CIFAR10(root='data', 
                         train=True, 
                         download=True, 
                         transform=transform                
                        )


# In[21]:


# datasets의 CIFAR10 데이터셋 로드 (test 데이터셋)
test = datasets.CIFAR10(root='data', 
                        train=False, 
                        download=True, 
                        transform=transform
                       )


# 아래 변환된 통계량을 보면, train 셋의 평균은 거의 `(0, 0, 0)`에 수렴하는 것을 확인할 수 있습니다. (이는 train 셋을 기준으로 변환했기 때문입니다.)

# In[22]:


print_stats(train)
print('==='*10)
print_stats(test)

