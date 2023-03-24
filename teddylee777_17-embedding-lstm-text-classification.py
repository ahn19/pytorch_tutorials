#!/usr/bin/env python
# coding: utf-8

# ## 데이터프레임(DataFrame) 커스텀 데이터셋 클래스
# 
# `torchtext.legacy.data.Dataset`을 확장하여 DataFrame을 바로 `BucketIterator`로 변환할 수 있습니다.

# In[1]:


import urllib
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 123

# bbc-text.csv 데이터셋 다운로드
url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
urllib.request.urlretrieve(url, 'bbc-text.csv')

# 데이터프레임을 로드 합니다.
df = pd.read_csv('bbc-text.csv')

# 컬럼명은 text / label 로 변경합니다
df = df.rename(columns={'category': 'label'})
df


# In[2]:


# train / validation 을 분할 합니다.
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)


# In[3]:


# train DataFrame
train_df.head()


# In[4]:


# validation DataFrame
val_df.head()


# In[5]:


# 필요한 모듈 import
import torch
from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer

# device 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


# `torchtext.legacy.data.Dataset`을 상속하여 데이터프레임을 로드할 수 있습니다.

# In[6]:


class DataFrameDataset(data.Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            # text, label 컬럼명은 필요시 변경하여 사용합니다
            label = row['label'] if not is_test else None
            text = row['text'] 
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, False, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


# In[7]:


# 토크나이저 정의 (다른 토크나이저로 대체 가능)
tokenizer = get_tokenizer('basic_english')


# 앞선 내용과 마찬가지로 `Field`를 구성합니다.

# In[8]:


TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)

# fiels 변수에 List(tuple(컬럼명, 변수)) 형식으로 구성 후 대입
fields = [('text', TEXT), ('label', LABEL)]


# In[9]:


# DataFrame의 Splits로 데이터셋 분할
train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=val_df)


# In[10]:


# 단어 사전 생성
TEXT.build_vocab(train_ds, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_ds)


# `BucketIterator`를 생성합니다.

# In[12]:


BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_ds, val_ds), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)


# In[13]:


# 1개 배치 추출
sample_data = next(iter(train_iterator))


# In[14]:


# text shape 출력 (batch_size, sequence_length)
sample_data.text.shape


# In[15]:


# label 출력 (batch)
sample_data.label


# ## Embedding Layer

# In[222]:


x = sample_data.text
x.shape


# In[223]:


# 단어 사전 개수 출력
NUM_VOCABS = len(TEXT.vocab)
print(f'Number of Vocabs: {NUM_VOCABS}')
# 개수 1000 + <unk> + <pad> : 총 1002개

EMBEDDING_DIM = 25
print(f'Embedding Dimension: {EMBEDDING_DIM}')

SEQ_LENGTH = 120
print(f'Sequence Length: {MAX_SEQ_LENGTH}')

print(f'Number of Batch Size: {BATCH_SIZE}')


# In[224]:


x.shape


# In[225]:


import torch.nn as nn

# Number of Vocabs, Embedding Dimension as an input
embedding = nn.Embedding(num_embeddings=NUM_VOCABS, 
                         embedding_dim=EMBEDDING_DIM, 
                         padding_idx=1, 
                         device=device)
embedding


# In[226]:


embedding_output = embedding(x)
embedding_output.shape
# batch_size, sequence_length, embedding_dim


# ## LSTM Output Shape

# `bidirectional=True` 인 경우에는 2 * `hidden_size`가 output의 마지막 shape로 출력됩니다.

# In[227]:


lstm = nn.LSTM(input_size=EMBEDDING_DIM, 
               hidden_size=64, 
               num_layers=2, 
               bidirectional=True,
               batch_first=False, 
               device=device
              )

lstm_output, (lstm_hidden, lstm_cell) = lstm(embedding_output)
lstm_output.shape
# output: sequence_length, batch_size, bidirectional(2)*hidden_size


# `bidirectional=False` 인 경우 1*`hidden_size`가 output의 마지막 shape로 출력됩니다.

# In[228]:


lstm = nn.LSTM(input_size=EMBEDDING_DIM, 
               hidden_size=64, 
               num_layers=2, 
               bidirectional=False,
               batch_first=False, 
               device=device
              )

lstm_output, (lstm_hidden, lstm_cell) = lstm(embedding_output)
lstm_output.shape
# output: sequence_length, batch_size, NO bidirectional(1)*hidden_size


# `batch_first=True`로 설정하는 경우
# 
# - 입력 텐서와 출력 텐서의 shape를 `(batch, seq, feature)` 형태를 가지도록 합니다. 만약 `False`로 설정된 경우에는 `(seq, batch, feature)`로 입출력이 됩니다. 일반적인 경우 batch가 첫 번째 shape에 위치하기 때문에 `batch_first=True`로 주로 설정합니다.
# - 하지만, `hidden state`, `cell state`에는 **해당 사항이 아닙니다**

# `batch_first=False`인 경우

# In[237]:


lstm = nn.LSTM(input_size=EMBEDDING_DIM, 
               hidden_size=64, 
               num_layers=2, 
               bidirectional=True,
               batch_first=False, 
               device=device
              )

# (32, 120, 25)
# sequence_length, batch_size, input_size

output, (hidden_state, cell_state) = lstm(embedding_output)
output.shape, hidden_state.shape, cell_state.shape
# output: sequence_length, batch_size, bidirectional(2)*hidden_size
# hidden_state: bidirectional(2)*num_layers, batch_size, hidden_size
# cell_state: bidirectional(2)*num_layers, batch_size, hidden_size


# `batch_first=True`인 경우

# In[231]:


lstm = nn.LSTM(input_size=EMBEDDING_DIM, 
               hidden_size=64, 
               num_layers=2, 
               bidirectional=True,
               batch_first=True, 
               device=device
              )

# (32, 120, 25)
# batch_size, sequence_length, input_size

output, (hidden_state, cell_state) = lstm(embedding_output)
output.shape, hidden_state.shape, cell_state.shape
# output: batch_size, sequence_length, bidirectional(2)*hidden_size
# hidden_state: bidirectional(2)*num_layers, batch_size, hidden_size
# cell_state: bidirectional(2)*num_layers, batch_size, hidden_size


# ## 정석 코딩!!

# 입력: `embedding_output`

# In[241]:


print(f'embedding_output.shape: {embedding_output.shape}')
# batch_size, sequence_length, embedding_dim


# In[239]:


lstm = nn.LSTM(input_size=EMBEDDING_DIM, 
               hidden_size=64, 
               num_layers=2, 
               bidirectional=True,
               batch_first=True, 
               device=device
              )
# input shape
# hidden_state_input: bidirectional(2)*num_layers, batch_size, hidden_size
# cell_state_input: bidirectional(2)*num_layers, batch_size, hidden_size
h_0 = torch.zeros(2*2, BATCH_SIZE, 64).to(device)
c_0 = torch.zeros(2*2, BATCH_SIZE, 64).to(device)

# 아래는 에러 발생의 예시
# h_0 = torch.zeros(BATCH_SIZE, 2*2, 64).to(device)
# c_0 = torch.zeros(BATCH_SIZE, 2*2, 64).to(device)

output, (hidden_state, cell_state) = lstm(embedding_output, (h_0, c_0))
output.shape, hidden_state.shape, cell_state.shape
# output: batch_size, sequence_length, bidirectional(2)*hidden_size
# hidden_state: bidirectional(2)*num_layers, batch_size, hidden_size
# cell_state: bidirectional(2)*num_layers, batch_size, hidden_size


# 가장 마지막 Sequence의 output을 가져옵니다.

# In[236]:


lstm_output[:, -1, :].shape


# ## 모델 생성

# In[211]:


from tqdm import tqdm  # Progress Bar 출력
import numpy as np
import torch.nn as nn
import torch.optim as optim


class TextClassificationModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_size, num_layers, seq_length, drop_prob=0.15):
        super(TextClassificationModel, self).__init__()
        self.num_classes = num_classes 
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim)
        
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            bidirectional=True,
                           )
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, hidden_and_cell):
        x = self.embedding(x)
        output, (h, c) = self.lstm(x, hidden_and_cell)
        h = output[:, -1, :]
        o = self.dropout(h)
        o = self.relu(self.fc(o))
        o = self.dropout(o)
        return self.output(o)


# In[212]:


config = {
    'num_classes': 5, 
    'vocab_size': NUM_VOCABS,
    'embedding_dim': 30, 
    'hidden_size': 64, 
    'num_layers': 2, 
    'seq_length': 120, 
}

model = TextClassificationModel(**config)
model.to(device)


# In[213]:


# loss 정의: CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저 정의: bert.paramters()와 learning_rate 설정
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# In[214]:


def model_train(model, data_loader, loss_fn, optimizer, config, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()
    
    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_loss = 0
    corr = 0
    counts = 0
    total_counts = 0
    
    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)
    
    # mini-batch 학습을 시작합니다.
    for idx, data in enumerate(prograss_bar):
        # text, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
        text = data.text.to(device)
        label = data.label.to(device)
        label.sub_(1)
        
        # 누적 Gradient를 초기화 합니다.
        optimizer.zero_grad()
        
        initial_hidden = torch.zeros(2*config['num_layers'], len(text), config['hidden_size']).to(device)
        initial_cell = torch.zeros(2*config['num_layers'], len(text), config['hidden_size']).to(device)

        # Forward Propagation을 진행하여 결과를 얻습니다.
        output = model(text, (initial_hidden, initial_cell))
        
        # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
        loss = loss_fn(output, label)
        
        # 오차역전파(Back Propagation)을 진행하여 미분 값을 계산합니다.
        loss.backward()
        
        # 계산된 Gradient를 업데이트 합니다.
        optimizer.step()
        
        # output의 max(dim=1)은 max probability와 max index를 반환합니다.
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
        _, pred = output.max(dim=1)
        
        # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
        # 합계는 corr 변수에 누적합니다.
        corr += pred.eq(label).sum().item()
        counts += label.size(0)
        
        # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
        # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
        # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
        running_loss += (loss.item() * label.size(0))
        
        total_counts += label.size(0)
        
        # 프로그레스바에 학습 상황 업데이트
        prograss_bar.set_description(f"training loss: {running_loss/total_counts:.5f}, training accuracy: {corr / counts:.5f}")
        
    # 누적된 정답수를 전체 개수로 나누어 주면 정확도가 산출됩니다.
    acc = corr / total_counts
    
    # 평균 손실(loss)과 정확도를 반환합니다.
    # train_loss, train_acc
    return running_loss / total_counts, acc


# In[215]:


def model_evaluate(model, data_loader, loss_fn, config, device):
    # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다. 
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
    model.eval()
    
    # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요합니다.
    with torch.no_grad():
        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        corr = 0
        running_loss = 0
        total_counts = 0
        
        # 배치별 evaluation을 진행합니다.
        for data in data_loader:
            # text, label 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)
            text = data.text.to(device)
            label = data.label.to(device)
            label.data.sub_(1)
            
            initial_hidden = torch.zeros(2*config['num_layers'], len(text), config['hidden_size']).to(device)
            initial_cell = torch.zeros(2*config['num_layers'], len(text), config['hidden_size']).to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출합니다.
            output = model(text, (initial_hidden, initial_cell))
            
            # output의 max(dim=1)은 max probability와 max index를 반환합니다.
            # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
            _, pred = output.max(dim=1)
            
            # pred.eq(lbl).sum() 은 정확히 맞춘 label의 합계를 계산합니다. item()은 tensor에서 값을 추출합니다.
            # 합계는 corr 변수에 누적합니다.
            corr += torch.sum(pred.eq(label)).item()
            
            # loss 값은 1개 배치의 평균 손실(loss) 입니다. img.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 img.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            running_loss += loss_fn(output, label).item() * label.size(0)
            
            total_counts += label.size(0)
        
        # validation 정확도를 계산합니다.
        # 누적한 정답숫자를 전체 데이터셋의 숫자로 나누어 최종 accuracy를 산출합니다.
        acc = corr / total_counts
        
        # 결과를 반환합니다.
        # val_loss, val_acc
        return running_loss / total_counts, acc


# In[216]:


# 최대 Epoch을 지정합니다.
num_epochs = 50

# checkpoint로 저장할 모델의 이름을 정의 합니다.
model_name = 'LSTM-Text-Classification'

min_loss = np.inf

# Epoch 별 훈련 및 검증을 수행합니다.
for epoch in range(num_epochs):
    # Model Training
    # 훈련 손실과 정확도를 반환 받습니다.
    train_loss, train_acc = model_train(model, train_iterator, loss_fn, optimizer, config, device)

    # 검증 손실과 검증 정확도를 반환 받습니다.
    val_loss, val_acc = model_evaluate(model, test_iterator, loss_fn, config, device)   
    
    # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(model.state_dict(), f'{model_name}.pth')
    
    # Epoch 별 결과를 출력합니다.
    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

