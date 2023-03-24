
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# mean=[0.485, 0.456, 0.406]과 std=[0.229, 0.224, 0.225]를 쓰는 이유는?
# https://velog.io/@neos960518/PyTorch%EC%97%90%EC%84%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%97%90-%EB%8C%80%ED%95%B4-normalize%EB%A5%BC-%ED%95%A0-%EB%95%8C-mean0.485-0.456-0.406%EA%B3%BC-std0.229-0.224-0.225%EB%A5%BC-%EC%93%B0%EB%8A%94-%EC%9D%B4%EC%9C%A0%EB%8A%94
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
# mean과 std 값으로 뭔가 이상한 값들이 세팅되어 있다. 많은 사용자들이 이 값들을 쓰는 이유가 무엇인지
# 제대로 모르고 그냥 쓰고 있는데, 사실 이 값들은 많은 Vision 모델들의 pretraining에 사용된
# ImageNet 데이터셋의 학습 시에 얻어낸 값들이다. ImageNet 데이터셋은 질 좋은 이미지들을 다량 포함하고
# 있기에 이런 데이터셋에서 얻어낸 값이라면 어떤 이미지 데이터 셋에서도 잘 작동할 것이라는 가정하에
# 이 값들을 기본 값으로 세팅해 놓은 것이다.

data_dir = 'data/hymenoptera_data'
a1 = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
a2 = data_transforms['train']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
# 'An attempt has been made to start a new process before ... freeze_support()..' 에러가
# 발생해서  아래 링크 조언대로 'num_workers=0'으로 설정
# https://stackoverflow.com/questions/64348346/errorruntimeerror-an-attempt-has-been-made-to-start-a-new-process-before-the-c
# https://aigong.tistory.com/136

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ### Visualize a few images

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
b1 = dataloaders['train']
# dataiter = iter(b1)
# images, labels = next(dataiter)
inputs, classes = next(iter(dataloaders['train']))
# 'An attempt has been made to start a new process before ... freeze_support()..' 에러가
# 발생해서  아래 링크 조언대로 'num_workers=0'으로 설정
# https://stackoverflow.com/questions/64348346/errorruntimeerror-an-attempt-has-been-made-to-start-a-new-process-before-the-c
# https://aigong.tistory.com/136

img0 = inputs[0]
np_arr = img0.cpu().detach().numpy()

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# ## Training the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    model_state_dict = model.state_dict()
    best_model_wts = copy.deepcopy(model.state_dict())
    # 깊은 복사는 내부에 객체들까지 모두 새롭게 copy 되는 것입니다.

    # state_dict란 torch.nn.Module에서 모델로 학습할 때 각 layer마다 텐서로 매핑되는 매개변수(예를 들어 가
    # 중치, 편향과 같은)를 python dictionary 타입으로 저장한 객체이다. 학습 가능한 매개변수를 갖는 계층만이
    # 모델의 state_dict에 항목을 가진다.
    # torch.optim 또한 옵티마이저의 상태 뿐만 아니라 사요된 하이퍼 매개변수 정보다 포함된 state_dict를 갖는다.
    # 한마디로 모델의 구조에 맞게 각 레이어마다의 매개변수를 tensor형태로 매핑해서 dictionary형태로 저장하는 것이다.
    # 왜 그냥 torch.save()로 저장하는 것이 아니라 state_dict()로 저장하는지는 잘 모르겠지만...
    # 조금 다르게 저장하고 불러오는 느낌이다.
    # https://everywhere-data.tistory.com/48
    # 아래의 글에서처럼 torch.save(the_model.state_dict(), PATH) 로 저장하는 게 좋다.
    # torch.save(the_model, PATH)로 저장하게 되면 아래와 같은 문제가 있다고 한다.
    # '그러나 이 경우 직렬화된 데이터는 사용된 특정 클래스 및 정확한 디렉토리 구조에 바인딩되므로
    # 다른 프로젝트에서 사용하거나 일부 심각한 리팩터링 후에 다양한 방식으로 중단될 수 있습니다.'
    # https://stackoverflow.com/a/43819235/18525539

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
                # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
                # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
                # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
                # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
                # https://stackoverflow.com/a/51433411/18525539
            else:
                model.eval()   # Set model to evaluate mode
                # model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
                # https://stackoverflow.com/a/51433411/18525539

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를
                # 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기
                # 위해 반복할 때마다 명시적으로 0으로 설정합니다.

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # torch.no_grad() 와 torch.set_grad_enabled(False)은 사실상 같다.
                    # no_grad 소스코드를 보면 실제로 클래스 내부에서 torch.set_grad_enabled를 사용하고 있음을 알 수 있습니다.
                    # https://stackoverflow.com/questions/53447345/pytorch-set-grad-enabledfalse-vs-with-no-grad
                    # torch.no_grad()은 autograd(미분 계산하는 기능)를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
                    # torch.no_grad()는 autograd engine을 꺼버린다. 이 말은 더 이상 자동으로 gradient를 트래킹하지 않는다는 말이 된다.
                    # 그러면 이런 의문이 들 수 있다. loss.backward()를 통해 backpropagation을 진행하지 않는다면 뭐
                    # gradient를 게산하든지 말든지 큰 상관이 없는 것이 아닌가?
                    # 맞는 말이다. torch.no_grad()의 주된 목적은 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
                    # 사실상 어짜피 안쓸 gradient인데 inference시에 굳이 계산할 필요가 없지 않은가?
                    # 그래서 일반적으로 inference를 진행할 때는 torch.no_grad() with statement로 감싼다는 사실을 알면 된다.
                    # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() # loss함수에서 모든 변수에 대해 미분을 계산한다.
                        optimizer.step() # 역전파 단계에서 수집된 gradients로 매개변수를 업데이트 한다

                # statistics
                a1 = loss.item()
                a2 = inputs.size(0)
                a3 = preds == labels.data
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step() # 매 epoch마다 scheduler.step()을 호출해서 learning rate 계수를
                # 감소시키는 것 같다.
                # If you don’t call it, the learning rate won’t be changed and stays at the initial value.
                # https://discuss.pytorch.org/t/what-does-scheduler-step-do/47764/2
                # PyTorch 1.1.0 이전에는 옵티마이저 업데이트 전에 학습 속도 스케줄러가 호출될 것으로 예상되었습니다.
                # 1.1.0에서는 이 동작을 BC를 깨는 방식으로 변경했습니다. 옵티마이저의 업데이트(optimizer.step() 호출) 전에
                # 학습률 스케줄러(scheduler.step() 호출)를 사용하는 경우 학습률 일정의 첫 번째 값을 건너뜁니다.
                # Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before
                # the optimizer’s update; 1.1.0 changed this behavior in a BC-breaking way. If you use
                # the learning rate scheduler (calling scheduler.step()) before the optimizer’s update
                # (calling optimizer.step()), this will skip the first value of the learning rate schedule.
                # https://pytorch.org/docs/stable/optim.html

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ### Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    # Set model to training mode
    # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
    # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
    # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
    # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
    # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
    # https://stackoverflow.com/a/51433411/18525539
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        # torch.no_grad()은 autograd(미분 계산하는 기능)를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
        # torch.no_grad()는 autograd engine을 꺼버린다. 이 말은 더 이상 자동으로 gradient를 트래킹하지 않는다는 말이 된다.
        # 그러면 이런 의문이 들 수 있다. loss.backward()를 통해 backpropagation을 진행하지 않는다면 뭐
        # gradient를 게산하든지 말든지 큰 상관이 없는 것이 아닌가?
        # 맞는 말이다. torch.no_grad()의 주된 목적은 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
        # 사실상 어짜피 안쓸 gradient인데 inference시에 굳이 계산할 필요가 없지 않은가?
        # 그래서 일반적으로 inference를 진행할 때는 torch.no_grad() with statement로 감싼다는 사실을 알면 된다.
        # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            a1 = inputs.size()
            for j in range(inputs.size()[0]):
                images_so_far += 1
                a2 = num_images // 2
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                a3 = preds[j]
                a4 = class_names[preds[j]]
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
                    # Set model to training mode
                    # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
                    # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
                    # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
                    # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
                    # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
                    # https://stackoverflow.com/a/51433411/18525539
                    return
        model.train(mode=was_training)

# ## Finetuning the convnet
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
model_ft_parameters = list(model_ft.parameters())
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# ### Train and evaluate
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)

visualize_model(model_ft)

# ## ConvNet as fixed feature extractor

model_conv = torchvision.models.resnet18(pretrained=True)
model_conv_parameters = list(model_conv.parameters())
cnt = 0
for param in model_conv.parameters():
    param.requires_grad = False
    cnt = cnt + 1

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
model_conv_fc_parameters = list(model_conv.fc.parameters())
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


# ### Train and evaluate
# model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=2)
visualize_model(model_conv)

plt.ioff()
plt.show()


# ## Further Learning












































