#!/usr/bin/env python
# coding: utf-8

# # torchtext 튜토리얼

# ## 샘플 데이터셋 다운로드

# In[43]:


import urllib

url = 'https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv'
urllib.request.urlretrieve(url, 'bbc-text.csv')


# Pandas로 데이터 로드 및 출력

# In[1]:


import pandas as pd

df = pd.read_csv('bbc-text.csv')
df


# ## 토크나이저 생성

# In[2]:


from torchtext.data.utils import get_tokenizer


# tokenizer의 타입으로는 `basic_english`, `spacy`, `moses`, `toktok`, `revtok`, `subword` 이 있습니다.
# 
# 다만, 이 중 몇개의 타입은 추가 패키지가 설치되어야 정상 동작합니다.

# In[3]:


tokenizer = get_tokenizer('basic_english', language='en')
tokenizer("I'd like to learn torchtext")


# 토큰 타입을 지정하면 그에 맞는 tokenizer를 반환하는 함수를 생성한 뒤 원하는 타입을 지정하여 tokenizer를 생성할 수 있습니다.

# In[4]:


def generate_tokenizer(tokenizer_type, language='en'):
    return get_tokenizer(tokenizer_type, language=language)


# `basic_english`를 적용한 경우

# In[5]:


tokenizer = generate_tokenizer('basic_english')
tokenizer("I'd like to learn torchtext")


# `toktok`을 적용한 경우

# In[6]:


tokenizer = generate_tokenizer('toktok')
tokenizer("I'd like to learn torchtext")


# In[7]:


from nltk.tokenize import word_tokenize

word_tokenize("I'd like to learn torchtext")


# ## 필드(Field) 정의

# In[8]:


from torchtext.legacy import data


# `torchtext.legacy.data.Field` 
# - `Field` 클래스는 `Tensor`로 변환하기 위한 지침과 함께 데이터 유형을 정의합니다. 
# - `Field` 객체는 `vocab` 개체를 보유합니다.
# - `Field` 객체는 토큰화 방법, 생성할 Tensor 종류와 같이 데이터 유형을 수치화하는 역할을 수행합니다.

# In[9]:


TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자 화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)


# `fields` 변수에 dictionary를 생성합니다.
# - `key`: 읽어 들여올 파일의 열 이름을 지정합니다.
# - `value`: (`문자열`, `data.Field`) 형식으로 지정합니다. 여기서 지정한 문자열이 나중에 생성된 data의 변수 이름으로 생성됩니다.
# 
# (참고) fields에 `[('text', TEXT), ('label', LABEL)]` 와 같이 생성하는 경우도 있습니다. 컬러명 변경이 필요하지 않은 경우는 `List(tuple(컬럼명, 변수))`로 생성할 수 있습니다.

# In[10]:


fields = {
    'text': ('text', TEXT), 
    'category': ('label', LABEL)
}


# ## 데이터셋 로드 및 분할

# `TabularDataset` 클래스는 정형 데이터파일로부터 직접 데이터를 읽을 때 유용합니다.
# 
# 지원하는 파일 형식은 `CSV`, `JSON`, `TSV` 을 지원합니다.

# In[11]:


import random
from torchtext.legacy.data import TabularDataset

SEED = 123

dataset = TabularDataset(path='bbc-text.csv',  # 파일의 경로
                         format='CSV',         # 형식 지정
                         fields=fields,        # 이전에 생성한 field 지정
#                          skip_header=True    # 첫 번째 행은 컬러명이므로 skip
                        )        


# 이전에 생성한 `dataset` 변수로 train / test 데이터셋을 분할 합니다.

# In[12]:


train_data, test_data = dataset.split(split_ratio=0.8,               # 분할 비율
                                      stratified=True,               # stratify 여부
                                      strata_field='label',          # stratify 대상 컬럼명
                                      random_state=random.seed(SEED) # 시드
                                     )


# In[13]:


# 생성된 train / test 데이터셋의 크기를 출력 합니다.
len(train_data), len(test_data)


# ## 단어 사전 생성

# In[14]:


TEXT.build_vocab(train_data, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_data)


# In[15]:


NUM_VOCABS = len(TEXT.vocab.stoi)
NUM_VOCABS


# In[16]:


TEXT.vocab.freqs.most_common(10)


# `TEXT.vocab.stoi`는 문자열을 index로, `TEXT.vocab.itos`는 index를 문자열로 변환합니다.

# In[17]:


TEXT.vocab.stoi


# In[18]:


# string to index
print(TEXT.vocab.stoi['this'])
print(TEXT.vocab.stoi['pretty'])
print(TEXT.vocab.stoi['original'])

print('==='*10)

# index to string
print(TEXT.vocab.itos[14])
print(TEXT.vocab.itos[194])
print(TEXT.vocab.itos[237])


# ## 버킷 이터레이터 생성
# 
# - `BucketIterator` 의 주된 역할은 데이터셋에 대한 배치 구성입니다.

# In[19]:


import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),     # dataset
    sort=False,
    repeat=False,
    batch_size=BATCH_SIZE,       # 배치사이즈
    device=device)               # device 지정


# 1개의 배치를 추출합니다.

# In[20]:


# 1개의 batch 추출
sample_data = next(iter(train_iterator))


# `text` 의 shape 를 확인합니다.

# In[21]:


# batch_size, sequence_length
sample_data.text.shape


# In[22]:


len(sample_data.text)


# In[23]:


sample_data.label.size(0)


# `label` 의 shape 를 확인합니다.

# In[24]:


# batch_size
sample_data.label.shape


# In[25]:


# label을 출력합니다.
sample_data.label


# 아래에서 확인할 수 있듯이 `<unk>` 토큰 때문에 카테고리의 개수가 5개임에도 불구하고 index는 0번부터 5번까지 맵핑되어 있습니다.

# In[26]:


LABEL.vocab.stoi


# 따라서, 0번을 무시해주기 위해서는 배치 학습시 다음과 같이 처리해 줄 수 있습니다.
# 
# 1을 subtract 해줌으로써 0~4번 index로 조정해 주는 것입니다.

# In[27]:


sample_data.label.sub_(1)


# ## 데이터프레임(DataFrame) 커스텀 데이터셋 클래스
# 
# `torchtext.legacy.data.Dataset`을 확장하여 DataFrame을 바로 `BucketIterator`로 변환할 수 있습니다.

# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 123

# 데이터프레임을 로드 합니다.
df = pd.read_csv('bbc-text.csv')

# 컬럼명은 text / label 로 변경합니다
df = df.rename(columns={'category': 'label'})
df


# In[29]:


# train / validation 을 분할 합니다.
train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)


# In[30]:


# train DataFrame
train_df.head()


# In[31]:


# validation DataFrame
val_df.head()


# In[32]:


# 필요한 모듈 import
import torch
from torchtext.legacy import data
from torchtext.data.utils import get_tokenizer

# device 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


# `torchtext.legacy.data.Dataset`을 상속하여 데이터프레임을 로드할 수 있습니다.

# In[33]:


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


# In[34]:


# 토크나이저 정의 (다른 토크나이저로 대체 가능)
tokenizer = get_tokenizer('basic_english')


# 앞선 내용과 마찬가지로 `Field`를 구성합니다.

# In[35]:


TEXT = data.Field(sequential=True,    # 순서를 반영
                  tokenize=tokenizer, # tokenizer 지정
                  fix_length=120,     # 한 문장의 최대 길이 지정
                  lower=True,         # 소문자화
                  batch_first=True)   # batch 를 가장 먼저 출력


LABEL = data.Field(sequential=False)

# fiels 변수에 List(tuple(컬럼명, 변수)) 형식으로 구성 후 대입
fields = [('text', TEXT), ('label', LABEL)]


# In[36]:


# DataFrame의 Splits로 데이터셋 분할
train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=val_df)


# In[37]:


# 단어 사전 생성
TEXT.build_vocab(train_ds, 
                 max_size=1000,             # 최대 vocab_size 지정 (미지정시 전체 단어사전 개수 대입)
                 min_freq=5,                # 최소 빈도 단어수 지정
                 vectors='glove.6B.100d')   # 워드임베딩 vector 지정, None으로 지정시 vector 사용 안함

LABEL.build_vocab(train_ds)


# In[38]:


# 단어 사전 개수 출력
NUM_VOCABS = len(TEXT.vocab)
NUM_VOCABS
# 개수 1000 + <unk> + <pad> : 총 1002개


# `BucketIterator`를 생성합니다.

# In[39]:


BATCH_SIZE = 32

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_ds, val_ds), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)


# In[40]:


# 1개 배치 추출
sample_data = next(iter(train_iterator))


# In[41]:


# text shape 출력 (batch_size, sequence_length)
sample_data.text.shape


# In[42]:


# label 출력 (batch)
sample_data.label

