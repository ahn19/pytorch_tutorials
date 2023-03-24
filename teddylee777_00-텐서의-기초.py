#!/usr/bin/env python
# coding: utf-8

# ## 스칼라, 벡터, 메트릭스, 텐서

# 아래의 그림에서 잘 설명되어 있듯이
# 
# - 단일 값은 `Scalar`
# - 1D(1차원) 데이터는 `Vector`
# - 2D(행렬) 데이터는 `Matrix`
# - 3차원 이상의 데이터는 `Tensor`

# In[16]:


from IPython.display import Image

Image(url='https://mblogthumb-phinf.pstatic.net/MjAyMDA1MjVfMTM0/MDAxNTkwMzc4MTY4MDQy.iOzxIfhew8Bsto7uqNW3QYj-k9bysF775jXYLECD6bwg.uMJ87NPURvklkXF2TXFnygaSnc32erHm_mXbnKgvO24g.PNG.nabilera1/image.png?type=w800')


# ## Tensor의 특성
# 
# - `torch.Tensor`는 단일 데이터 타입(single data type)을 가집니다.
# - `torch.Tensor` 간의 연산은 **같은 데이터타입**일 경우에만 가능합니다.
# - `Numpy`의 배열 연산으로 수행할 수 있는 내용도, GPU를 활용하여 빠르게 학습하려는 경우 `torch.Tensor`로 변환할 수 있습니다.

# ## 필요한 모듈 import

# In[4]:


# 모듈 import
import torch
import numpy as np


# pytorch version 체크

# In[5]:


print(torch.__version__)


# ## Numpy로부터 Tensor 변환
# 
# - 방법1. `torch.from_numpy()`
# - 방법2. `torch.as_tensor()`
# - 방법3. `torch.tensor()`

# In[6]:


# 샘플 데이터 생성
arr = np.array([1, 3, 5, 7, 9])
print(arr)
print(arr.dtype)
print(type(arr))


# `torch.from_numpy()`는 `torch.as_tensor()`와 동일

# ### `torch.from_numpy()` - sharing

# In[12]:


t1 = torch.from_numpy(arr)
print(t1) # 출력
print(t1.dtype)  # dtype은 데이터 타입
print(t1.type()) # type()은 텐서의 타입
print(type(t1))  # t1 변수 자체의 타입


# ### `torch.as_tensor()` - sharing

# In[13]:


t2 = torch.as_tensor(arr)
print(t2) # 출력
print(t2.dtype)  # dtype은 데이터 타입
print(t2.type()) # type()은 텐서의 타입
print(type(t2))  # t2 변수 자체의 타입


# In[14]:


# numpy array의 0번 index를 999로 값 변환
arr[0] = 999


# In[17]:


# t1, t2 출력
print(f't1: {t1}')
print(f't2: {t2}')


# `torch.from_numpy()`와 `torch.as_tensor()` 로 `numpy array`의 요소를 수정하게 되면 해당 numpy array로부터 **생성된 tensor의 요소의 값이 변하는 것**을 확인할 수 있습니다.
# 
# 이러한 현상은 `torch.from_numpy()`와 `torch.as_tensor()` 모두 **sharing** 하기 때문입니다.

# ### `torch.tensor()` - copying

# In[18]:


# 샘플 데이터 초기화
arr = np.array([1, 3, 5, 7, 9])
print(arr)
print(arr.dtype)
print(type(arr))


# In[19]:


t3 = torch.tensor(arr)
print(t3) # 출력
print(t3.dtype)  # dtype은 데이터 타입
print(t3.type()) # type()은 텐서의 타입
print(type(t3))  # t3 변수 자체의 타입


# In[20]:


# numpy array의 0번 index를 999로 값 변환
arr[0] = 999


# In[21]:


# t3 출력
print(f't3: {t3}')


# `torch.tensor()`로 numpy array를 변환시 **sharing**이 아닌 **copying**하기 때문에 원본 **numpy array의 요소가 변하더라고 tensor에 영향을 끼치지 않음을 확인**할 수 있습니다.

# ## 텐서의 데이터 타입
# 
# - [pytorch 도큐먼트 링크](https://pytorch.org/docs/stable/tensors.html)

# <table style="display: inline-block">
# <tr><th>TYPE</th><th>NAME</th><th>EQUIVALENT</th><th>TENSOR TYPE</th></tr>
# <tr><td>32-bit integer (signed)</td><td>torch.int32</td><td>torch.int</td><td>IntTensor</td></tr>
# <tr><td>64-bit integer (signed)</td><td>torch.int64</td><td>torch.long</td><td>LongTensor</td></tr>
# <tr><td>16-bit integer (signed)</td><td>torch.int16</td><td>torch.short</td><td>ShortTensor</td></tr>
# <tr><td>32-bit floating point</td><td>torch.float32</td><td>torch.float</td><td>FloatTensor</td></tr>
# <tr><td>64-bit floating point</td><td>torch.float64</td><td>torch.double</td><td>DoubleTensor</td></tr>
# <tr><td>16-bit floating point</td><td>torch.float16</td><td>torch.half</td><td>HalfTensor</td></tr>
# <tr><td>8-bit integer (signed)</td><td>torch.int8</td><td></td><td>CharTensor</td></tr>
# <tr><td>8-bit integer (unsigned)</td><td>torch.uint8</td><td></td><td>ByteTensor</td></tr></table>
