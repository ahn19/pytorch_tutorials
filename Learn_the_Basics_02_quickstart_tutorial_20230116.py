
# get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# ## Creating Models
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

a1 = list(model.parameters())
a2 = list(model.parameters())[0]
a3 = list(model.parameters())[0].size()
a4 = list(model.parameters())[1]

# ## Optimizing the Model Parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # model.train()은 모델을 교육하고 있음을 모델에 알립니다.
    # 이는 교육 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 레이어에 정보를 제공하는 데 도움이 됩니다.
    # 예를 들어 train 모드에서 BatchNorm은 각각의 새 배치에서 이동 평균을 업데이트합니다.
    # 반면 test 모드의 경우 BatchNorm은 이러한 업데이트가 중지됩니다.
    # model.train()은 training중임을 알리는 것이고, model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
    # https://stackoverflow.com/a/51433411/18525539
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X) # Forward Propagation을 진행하여 결과를 얻습니다.
        loss = loss_fn(pred, y) # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.

        # Backpropagation
        optimizer.zero_grad() # optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를
        # 재설정합니다. 기본적으로 변화도는 더해지기(add up) 때문에 중복 계산을 막기
        # 위해 반복할 때마다 명시적으로 0으로 설정합니다.
        loss.backward() # loss함수에서 모든 변수에 대해 미분을 계산한다.
        optimizer.step() # 역전파 단계에서 수집된 gradients로 매개변수를 업데이트 한다

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader) # 샘플10000개/배치사이즈64 = 156.x
    model.eval() # model.eval() 또는 model.train(mode=False)를 호출하여 테스트 중임을 알립니다.
    # https://stackoverflow.com/a/51433411/18525539
    test_loss, correct = 0, 0
    with torch.no_grad():
        # torch.no_grad()은 autograd(미분 계산하는 기능)를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
        # torch.no_grad()는 autograd engine을 꺼버린다. 이 말은 더 이상 자동으로 gradient를 트래킹하지 않는다는 말이 된다.
        # 그러면 이런 의문이 들 수 있다. loss.backward()를 통해 backpropagation을 진행하지 않는다면 뭐
        # gradient를 게산하든지 말든지 큰 상관이 없는 것이 아닌가?
        # 맞는 말이다. torch.no_grad()의 주된 목적은 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
        # 사실상 어짜피 안쓸 gradient인데 inference시에 굳이 계산할 필요가 없지 않은가?
        # 그래서 일반적으로 inference를 진행할 때는 torch.no_grad() with statement로 감싼다는 사실을 알면 된다.
        # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            a1 = loss_fn(pred, y).item()
            a2 = pred.argmax(1)
            a3 = pred.argmax(1) == y
            a4 = (pred.argmax(1) == y).type(torch.float)
            a5 = (pred.argmax(1) == y).type(torch.float).sum()
            a6 = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # 출력창에 6400개씩 출력된다. 왜냐면 배치사이즈 64 * 100개마다 print하므로
    test(test_dataloader, model, loss_fn)
print("Done!")

# ## Saving Models
a1 = model.state_dict()
a2 = model.state_dict()['linear_relu_stack.0.weight']
a3 = model.state_dict()['linear_relu_stack.0.bias']
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# ## Loading Models
model = NeuralNetwork() # re-creating the model structure
model.load_state_dict(torch.load("model.pth"))
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    # torch.no_grad()은 autograd(미분 계산하는 기능)를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
    # torch.no_grad()는 autograd engine을 꺼버린다. 이 말은 더 이상 자동으로 gradient를 트래킹하지 않는다는 말이 된다.
    # 그러면 이런 의문이 들 수 있다. loss.backward()를 통해 backpropagation을 진행하지 않는다면 뭐
    # gradient를 게산하든지 말든지 큰 상관이 없는 것이 아닌가?
    # 맞는 말이다. torch.no_grad()의 주된 목적은 autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함이다.
    # 사실상 어짜피 안쓸 gradient인데 inference시에 굳이 계산할 필요가 없지 않은가?
    # 그래서 일반적으로 inference를 진행할 때는 torch.no_grad() with statement로 감싼다는 사실을 알면 된다.
    # https://coffeedjimmy.github.io/pytorch/2019/11/05/pytorch_nograd_vs_train_eval/
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')






















































