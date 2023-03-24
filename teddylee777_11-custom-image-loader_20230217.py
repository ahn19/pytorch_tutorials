
# https://github.com/teddylee777/machine-learning/blob/master/02-PyTorch/11-custom-image-loader.ipynb
# https://teddylee777.github.io/pytorch/custom-image-loader/

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# `torchvision`의 `Image Transform` 에 대하여 생소하다면 **다음의 링크를 참고**해 주시기 바랍니다.
# - [torchvision의 transform으로 이미지 정규화하기(평균, 표준편차를 계산하여 적용)]
# (https://teddylee777.github.io/pytorch/torchvision-transform)
# - [PyTorch 이미지 데이터셋(Image Dataset) 구성에 관한 거의 모든 것!]
# (https://teddylee777.github.io/pytorch/dataset-dataloader)

# `개와 고양이` 데이터셋을 다운로드 받아서 `tmp` 폴더에 압축을 풀어 줍니다.
# [데이터셋 출처: 캐글, 마이크로소프트]
# (https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)
# 하단의 `code snippets`는 corrupted 된 이미지를 확인하고 제거하기 위한 코드 입니다.
# `Cats vs Dogs`데이터셋에도 원인 모를 이유 때문에 이미지 데이터가 corrupt된 파일이 2개가 존재합니다.
# 이렇게 corrupt 된 이미지를 `DataLoader`로 로드시 에러가 발생하기 때문에 전처리 때 미리 제거하도록 하겠습니다.

# ## 데이터셋 다운로드 및 압축 해제
# 이미지 데이터셋 다운로드
import urllib.request
import zipfile
import glob
import os
import random
from PIL import Image, UnidentifiedImageError,ImageFile

SEED = 123

# # 데이터셋을 다운로드 합니다.
# url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
# urllib.request.urlretrieve(url, 'cats_and_dogs.zip')
#
# # 다운로드 후 tmp 폴더에 압축을 해제 합니다.
# local_zip = 'cats_and_dogs.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('tmp/')
# zip_ref.close()

# 다운로드 받은 이미지에 대하여 정규성 검증을 수행합니다. (실제 Corrupted된 이미지가 2건이 있습니다.)
# 잘린 이미지 Load 시 경고 출력 안함
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 이미지 Validation을 수행하고 Validate 여부를 return 합니다.
def validate_image(filepath):
    try:
        # PIL.Image로 이미지 데이터를 로드하려고 시도합니다.
        img = Image.open(filepath).convert('RGB')
        img.load()
    except UnidentifiedImageError: # corrupt 된 이미지는 해당 에러를 출력합니다.
        print(f'Corrupted Image is found at: {filepath}')
        return False
    except (IOError, OSError): # Truncated (잘린) 이미지에 대한 에러를 출력합니다.
        print(f'Truncated Image is found at: {filepath}')
        return False
    else:
        return True

# image 데이터셋 root 폴더
root = 'tmp/PetImages' 
dirs = os.listdir(root)

for dir_ in dirs:
    folder_path = os.path.join(root, dir_)
    files = os.listdir(folder_path)
    
    images = [os.path.join(folder_path, f) for f in files]
    for img in images:
        valid = validate_image(img)
        if not valid:
            # corrupted 된 이미지 제거
            os.remove(img)

folders = glob.glob('tmp/PetImages/*')
print(folders)

# ## Train / Test 셋 분할
# train: test ratio. 0.2로 설정시 test set의 비율은 20%로 설정
test_size = 0.2

# train / test 셋의 파일을 나눕니다.
train_images = []
test_images = []

for folder in folders:
    label = os.path.basename(folder)
    files = sorted(glob.glob(folder + '/*'))

    # 각 Label별 이미지 데이터셋 셔플
    random.seed(SEED)
    random.shuffle(files)

    len_files = len(files)
    idx = int(len(files) * test_size)
    train = files[:-idx]
    test = files[-idx:]

    train_images.extend(train)
    test_images.extend(test)

# train, test 전체 이미지 셔플
random.shuffle(train_images)
random.shuffle(test_images)

# Class to Index 생성. {'Dog': 0, 'Cat': 1}
class_to_idx = {os.path.basename(f):idx for idx, f in enumerate(folders)}

# Label 생성
# train_labels = [f.split('/')[-2] for f in train_images]
# test_labels = [f.split('/')[-2] for f in test_images]
train_labels = [f.split('\\')[-2] for f in train_images]
test_labels = [f.split('\\')[-2] for f in test_images]

print('==='*10)
print(f'train images: {len(train_images)}')
print(f'train labels: {len(train_labels)}')
print(f'test images: {len(test_images)}')
print(f'test labels: {len(test_labels)}')

# ## Dataset 을 상속 받아 CustomImageDataset 구성
class CustomImageDataset(Dataset):
    def __init__(self, files, labels, class_to_idx, transform):
        super(CustomImageDataset, self).__init__()
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # file 경로
        file = self.files[idx]
        # PIL.Image로 이미지 로드
        img = Image.open(file).convert('RGB')
        # transform 적용
        img = self.transform(img)
        # label 생성
        lbl = self.class_to_idx[self.labels[idx]]
        # image, label return
        return img, lbl

# `Image Augmentation`을 적용 합니다.
# - [PyTorch Image Augmentation 도큐먼트](https://pytorch.org/vision/stable/transforms.html)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),          # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
    transforms.CenterCrop((224, 224)),      # 중앙 Crop
    transforms.RandomHorizontalFlip(0.5),   # 50% 확률로 Horizontal Flip
    transforms.ToTensor(), 
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
    transforms.ToTensor(), 
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
])

# train, test 데이터셋 생성
train_dataset = CustomImageDataset(train_images, train_labels, class_to_idx, train_transform)
test_dataset = CustomImageDataset(test_images, test_labels, class_to_idx, test_transform)

# `CustomImageDataset`은 `Dataset` 클래스를 상속받았기 때문에 DataLoader 안에 바로 대입할 수 있습니다.
# train, test 데이터 로더 생성 => 모델 학습시 입력하는 데이터셋
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

# 1개의 배치를 추출합니다.
images, labels = next(iter(train_loader))

# 이미지의 shape을 확인합니다. 224 X 224 RGB 이미지 임을 확인합니다.
images0_shape = images[0].shape

# `개와 고양이` 데이터셋 시각화
# - 총 2개의 class(강아지/고양이)로 구성된  사진 파일입니다.
import matplotlib.pyplot as plt

# ImageFolder의 속성 값인 class_to_idx를 할당
labels_map = {v:k for k, v in train_dataset.class_to_idx.items()}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 4

# 이미지를 출력합니다. RGB 이미지로 구성되어 있습니다.
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(images), size=(1,)).item()
    img, label = images[sample_idx], labels[sample_idx].item()
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    # 본래 이미지의 shape은 (3, 300, 300) 입니다.
    # 이를 imshow() 함수로 이미지 시각화 하기 위하여 (300, 300, 3)으로 shape 변경을 한 후 시각화합니다.
    plt.imshow(torch.permute(img, (1, 2, 0)))
plt.show()

# ## pre-trained 모델 로드
# CUDA 설정이 되어 있다면 `cuda`를! 그렇지 않다면 `cpu`로 학습합니다.
# (제 PC에는 GPU가 2대 있어서 `cuda:0`로 GPU 장비의 index를 지정해 주었습니다.
# 만약 다른 장비를 사용하고 싶다면 `cuda:1` 이런식으로 지정해 주면 됩니다)
# device 설정 (cuda:0 혹은 cpu)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# **pre-trained model**을 fine tuning 하여 Image Classification을 구현하도록 하겠습니다.
from torchvision import models # pretrained 모델을 가져오기 위한 import

# ResNet101 모델 생성
model = models.resnet101(pretrained=True) # pretrained=True 로 설정, pretrained=False로 설정되었을 경우 가중치는 가져오지 않습니다.
num_ftrs = model.fc.in_features
model_parameters = list(model.parameters())

# 그 밖의 활용 가능한 pretrained 모델
# - `models.alexnet(pregrained=True)`      # AlexNet
# - `models.vgg16(pretrained=True)`        # VGG16
# - `models.resnet18(pretrained=True)`     # ResNet18
# - `models.inception_v3(pretrained=True)` # Inception_V3

# 가중치를 Freeze 하여 학습시 업데이트가 일어나지 않도록 설정합니다.
for param in model.parameters():
    param.requires_grad = False  # 가중치 Freeze

import torch.nn as nn

# Fully-Connected Layer를 Sequential로 생성하여 VGG pretrained 모델의 'Classifier'에 연결합니다.
fc = nn.Sequential(
    nn.Linear(2048, 256), # VGG16 모델의 features의 출력이 7X7, 512장 이기 때문에 in_features=7*7*512 로 설정합니다.
    nn.ReLU(), 
    nn.Linear(256, 64), 
    nn.ReLU(), 
    nn.Linear(64, 2), # Cats vs Dogs 이진 분류이기 때문에 2로 out_features=2로 설정합니다.
)

model.fc = fc
model.to(device)
# 모델의 구조도 출력
print(model)

import torch.optim as optim
# 옵티마이저를 정의합니다. 옵티마이저에는 model.parameters()를 지정해야 합니다.
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# 왜 optim.Adam() 또는 optim.SGD()의 1번째 인자에 'model.parameters()'을 줘야할까? 아래 글에서처럼 결국
# 최적화할 대상을 지정해줘야 하기 때문인 것 같다.
# 'Optimizer를 구성하려면 최적화할 매개변수(모두 Variables여야 함)를 포함하는 iterable을 제공해야 합니다.
# 그런 다음 학습률, 가중치 감쇠 등과 같은 옵티마이저 관련 옵션을 지정할 수 있습니다.
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam([var1, var2], lr=0.0001)'
# To construct an Optimizer you have to give it an iterable containing the parameters
# (all should be Variable s) to optimize. Then, you can specify optimizer-specific options
# such as the learning rate, weight decay, etc.
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam([var1, var2], lr=0.0001)
# https://pytorch.org/docs/stable/optim.html
# 'optimizer = optim.Adam([var1, var2], lr=0.0001)' 과 같이 최적화할 변수를 따로 지정해줘도 된다.

# 손실함수(loss function)을 지정합니다. Multi-Class Classification 이기 때문에 CrossEntropy 손실을 지정하였습니다.
loss_fn = nn.CrossEntropyLoss()

# ## 훈련(Train)
from tqdm import tqdm  # Progress Bar 출력

def model_train(model, data_loader, loss_fn, optimizer, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()
    # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
    # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
    # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
    # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
    # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
    # https://stackoverflow.com/a/51433411/18525539
    
    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_loss = 0
    corr = 0
    
    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
    prograss_bar = tqdm(data_loader)
    
    # mini-batch 학습을 시작합니다.
    for img, lbl in prograss_bar:
        # image, label 데이터를 device에 올립니다.
        img, lbl = img.to(device), lbl.to(device)
        
        # 누적 Gradient를 초기화 합니다.
        optimizer.zero_grad()
        # optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를
        # 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기
        # 위해 반복할 때마다 명시적으로 0으로 설정합니다.
        
        # Forward Propagation을 진행하여 결과를 얻습니다.
        output = model(img)
        #  __call__ 메서드가 호출될 때 내부적으로 forward()가 호출된다
        #  __init__은 인스턴스 생성할 때 호출되고, __call__은 인스턴스 변수를 실행할 때 호출된다.
        # https://theonly1.tistory.com/3103
        
        # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
        loss = loss_fn(output, lbl)
        
        # 오차역전파(Back Propagation)을 진행하여 미분 값을 계산합니다.
        loss.backward() # loss함수에서 모든 변수에 대해 미분을 계산한다.
        
        # 계산된 Gradient를 업데이트 합니다.
        optimizer.step() # 역전파 단계에서 수집된 gradients로 매개변수를 업데이트 한다
        
        # output의 max(dim=1)은 max probability와 max index를 반환합니다.
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
        _, pred = output.max(dim=1)
        
        # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
        # 합계는 corr 변수에 누적합니다.
        c1 = pred.eq(lbl)
        c2 = pred.eq(lbl).sum()
        c3 = pred.eq(lbl).sum().item()
        corr += pred.eq(lbl).sum().item()
        
        # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
        # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
        # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
        loss_item = loss.item()
        img_size0 = img.size(0)
        running_loss += loss.item() * img.size(0)
        
    # 누적된 정답수를 전체 개수로 나누어 주면 정확도가 산출됩니다.
    acc = corr / len(data_loader.dataset)
    
    # 평균 손실(loss)과 정확도를 반환합니다.
    # train_loss, train_acc
    return running_loss / len(data_loader.dataset), acc


# ## 평가(Evaluate)
def model_evaluate(model, data_loader, loss_fn, device):
    # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다. 
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
    model.eval()
    # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
    # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
    # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
    # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
    # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
    # https://stackoverflow.com/a/51433411/18525539
    
    # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요합니다.
    with torch.no_grad():
        # torch.no_grad()은 autograd(미분 계산하는 기능)를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
        # torch.no_grad()는 autograd engine을 꺼버린다. 이 말은 더 이상 자동으로 gradient를 트래킹하지 않는다는 말이 된다.
        # 그러면 이런 의문이 들 수 있다. loss.backward()를 통해 backpropagation을 진행하지 않는다면 뭐
        # gradient를 게산하든지 말든지 큰 상관이 없는 것이 아닌가?
        # 맞는 말이다. torch.no_grad()의 주된 목적은 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
        # 사실상 어짜피 안쓸 gradient인데 inference시에 굳이 계산할 필요가 없지 않은가?
        # 그래서 일반적으로 inference를 진행할 때는 torch.no_grad() with statement로 감싼다는 사실을 알면 된다.
        # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/

        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        corr = 0
        running_loss = 0
        
        # 배치별 evaluation을 진행합니다.
        for img, lbl in data_loader:
            # device에 데이터를 올립니다.
            img, lbl = img.to(device), lbl.to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출합니다.
            output = model(img)
            
            # output의 max(dim=1)은 max probability와 max index를 반환합니다.
            # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
            _, pred = output.max(dim=1)
            
            # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
            # 합계는 corr 변수에 누적합니다.
            a1 = pred.eq(lbl)
            a2 = torch.sum(pred.eq(lbl)).item()
            corr += torch.sum(pred.eq(lbl)).item()
            
            # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            running_loss += loss_fn(output, lbl).item() * img.size(0)
        
        # validation 정확도를 계산합니다.
        # 누적한 정답숫자를 전체 데이터셋의 숫자로 나누어 최종 accuracy를 산출합니다.
        acc = corr / len(data_loader.dataset)
        
        # 결과를 반환합니다.
        # val_loss, val_acc
        return running_loss / len(data_loader.dataset), acc

# ## 모델 훈련(training) & 검증
# 최대 Epoch을 지정합니다.
num_epochs = 10
model_name = 'vgg16-pretrained'

min_loss = np.inf

# Epoch 별 훈련 및 검증을 수행합니다.
for epoch in range(num_epochs):
    # Model Training
    # 훈련 손실과 정확도를 반환 받습니다.
    train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)

    # 검증 손실과 검증 정확도를 반환 받습니다.
    val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)   
    
    # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(model.state_dict(), f'{model_name}.pth')
    
    # Epoch 별 결과를 출력합니다.
    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

# ## 저장한 가중치 로드 후 검증 성능 측정
# 모델에 저장한 가중치를 로드합니다.
model.load_state_dict(torch.load(f'{model_name}.pth'))

# 최종 검증 손실(validation loss)와 검증 정확도(validation accuracy)를 산출합니다.
final_loss, final_acc = model_evaluate(model, test_loader, loss_fn, device)
print(f'evaluation loss: {final_loss:.5f}, evaluation accuracy: {final_acc:.5f}')
































