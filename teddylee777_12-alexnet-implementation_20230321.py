
# https://teddylee777.github.io/pytorch/alexnet-implementation/
# https://github.com/teddylee777/machine-learning/blob/master/02-PyTorch/12-alexnet-implementation.ipynb

# ## AlexNet 구현
# AlexNet(2012) 의 PyTorch 구현 입니다. 논문에 대한 세부 인사이트는 생략하며, 오직 코드 구현만 다룹니다.
# - 논문 링크 [**(링크)**](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
# ![](https://sushscience.files.wordpress.com/2016/12/alexnet6.jpg)
# 출처: https://sushscience.wordpress.com/2016/12/04/understanding-alexnet/

# run() 함수 내에 구현한 이유는
# https://stackoverflow.com/questions/64348346/errorruntimeerror-an-attempt-has-been-made-to-start-a-new-process-before-the-c
# https://aigong.tistory.com/136
def run():

    # Hyper Parameter 설정
    IMAGE_SIZE = 227 # AlexNet의 이미지 입력 크기는 (3, 227, 227) 입니다.
    NUM_EPOCHS = 10
    LR = 0.0001 # Learning Rate

    MODEL_NAME = 'AlexNet'

    # ## 학습에 활용할 데이터셋 준비
    from torchvision import transforms
    #import sample_datasets as sd
    import teddylee777_sample_dataset_20230321 as sd

    from tqdm import tqdm  # Progress Bar 출력
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),          # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),      # 중앙 Crop
        transforms.RandomHorizontalFlip(0.5),   # 50% 확률로 Horizontal Flip
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),      # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
    ])

    # train_loader, test_loader = sd.cats_and_dogs(train_transform, test_transform)
    train_loader, test_loader = sd.cats_and_dogs1(train_transform, test_transform)

    torch.multiprocessing.freeze_support()
    # https://stackoverflow.com/questions/64348346/errorruntimeerror-an-attempt-has-been-made-to-start-a-new-process-before-the-c
    # 와 https://aigong.tistory.com/136

    # 1개의 배치를 추출합니다.
    images, labels = next(iter(train_loader))

    # 이미지의 shape을 확인합니다. 224 X 224 RGB 이미지 임을 확인합니다.
    images0shape = images[0].shape

    # ## AlexNet Architecture
    # ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-19-16-01-03.png)
    # 출처: https://www.datasciencecentral.com/alexnet-implementation-using-keras/
    # CUDA 설정이 되어 있다면 `cuda`를! 그렇지 않다면 `cpu`로 학습합니다.
    # (제 PC에는 GPU가 2대 있어서 `cuda:0`로 GPU 장비의 index를 지정해 주었습니다. 만약 다른 장비를 사용하고 싶다면 `cuda:1` 이런식으로 지정해 주면 됩니다)

    # device 설정 (cuda:0 혹은 cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    class AlexNet(nn.Module):
        def __init__(self):
            super(AlexNet, self).__init__()
            # Image input_size=(3, 227, 227)
            self.layers = nn.Sequential(
                # input_size=(96, 55, 55)
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4, padding=0),
                nn.ReLU(),
                # input_size=(96, 27, 27)
                nn.MaxPool2d(kernel_size=3, stride=2),
                # input_size=(256, 27, 27)
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
                nn.ReLU(),
                # input_size=(256, 13, 13)
                nn.MaxPool2d(kernel_size=3, stride=2),
                # input_size=(384, 13, 13)
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                # input_size=(384, 13, 13)
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                # input_size=(256, 13, 13)
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                # input_size=(256, 6, 6)
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features=256*6*6, out_features=4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096, out_features=1000), #out_features=2로 해야하는 거 아닌가?
            )

        def forward(self, x):
            x = self.layers(x)
            x = x.view(-1, 256*6*6)
            x = self.classifier(x)
            return x

    import torchsummary

    model = AlexNet()
    model.to(device)

    # AlexNet의 Image 입력 사이즈는 (3, 227, 227) 입니다.
    torchsummary.summary(model, input_size=(3, 227, 227), device='cuda')

    # 옵티마이저를 정의합니다. 옵티마이저에는 model.parameters()를 지정해야 합니다.
    optimizer = optim.Adam(model.parameters(), lr=LR)
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
    def model_train(model, data_loader, loss_fn, optimizer, device):
        # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
        model.train() # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
        # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
        # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
        # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
        # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
        # https://stackoverflow.com/a/51433411/18525539

        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        running_size = 0
        running_loss = 0
        corr = 0

        # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
        prograss_bar = tqdm(data_loader)

        # mini-batch 학습을 시작합니다.
        for batch_idx, (img, lbl) in enumerate(prograss_bar, start=1):
            # image, label 데이터를 device에 올립니다.
            img, lbl = img.to(device), lbl.to(device)

            # 누적 Gradient를 초기화 합니다.
            optimizer.zero_grad() # optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를
            # 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기
            # 위해 반복할 때마다 명시적으로 0으로 설정합니다.

            # Forward Propagation을 진행하여 결과를 얻습니다.
            output = model(img)

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
            b1 = pred.eq(lbl)
            b2 = pred.eq(lbl).sum()
            b3 = pred.eq(lbl).sum().item()
            corr += pred.eq(lbl).sum().item()

            # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            a1 = loss.item()
            a2 = img.size(0)
            a3 = loss.item() * img.size(0)
            running_loss += loss.item() * img.size(0)
            running_size += img.size(0)
            prograss_bar.set_description(f'[Training] loss: {running_loss / running_size:.4f}, accuracy: {corr / running_size:.4f}')

        # 누적된 정답수를 전체 개수로 나누어 주면 정확도가 산출됩니다.
        a1 = len(data_loader.dataset)
        acc = corr / len(data_loader.dataset)

        # 평균 손실(loss)과 정확도를 반환합니다.
        # train_loss, train_acc
        return running_loss / len(data_loader.dataset), acc

    # ## 평가(Evaluate)
    def model_evaluate(model, data_loader, loss_fn, device):
        # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다.
        # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
        model.eval() # model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.

        # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요합니다.
        with torch.no_grad():
            # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
            # torch.no_grad()은 autograd(미분 계산하는 기능)를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
            # torch.no_grad()는 autograd engine을 꺼버린다. 이 말은 더 이상 자동으로 gradient를 트래킹하지 않는다는 말이 된다.
            # 그러면 이런 의문이 들 수 있다. loss.backward()를 통해 backpropagation을 진행하지 않는다면 뭐
            # gradient를 게산하든지 말든지 큰 상관이 없는 것이 아닌가?
            # 맞는 말이다. torch.no_grad()의 주된 목적은 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
            # 사실상 어짜피 안쓸 gradient인데 inference시에 굳이 계산할 필요가 없지 않은가?
            # 그래서 일반적으로 inference를 진행할 때는 torch.no_grad() with statement로 감싼다는 사실을 알면 된다.
            # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/
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
    min_loss = np.inf

    # Epoch 별 훈련 및 검증을 수행합니다.
    for epoch in range(NUM_EPOCHS):
        # Model Training
        # 훈련 손실과 정확도를 반환 받습니다.
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)

        # 검증 손실과 검증 정확도를 반환 받습니다.
        val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)

        # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), f'{MODEL_NAME}.pth')
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

        # Epoch 별 결과를 출력합니다.
        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

    # ## 저장한 가중치 로드 후 검증 성능 측정
    # 모델에 저장한 가중치를 로드합니다.
    model.load_state_dict(torch.load(f'{MODEL_NAME}.pth'))

    # 최종 검증 손실(validation loss)와 검증 정확도(validation accuracy)를 산출합니다.
    final_loss, final_acc = model_evaluate(model, test_loader, loss_fn, device)
    print(f'evaluation loss: {final_loss:.5f}, evaluation accuracy: {final_acc:.5f}')

# if __name__ == '__main__': 한 이유는
# https://stackoverflow.com/questions/64348346/errorruntimeerror-an-attempt-has-been-made-to-start-a-new-process-before-the-c
# https://aigong.tistory.com/136
if __name__ == '__main__':
    run()








































