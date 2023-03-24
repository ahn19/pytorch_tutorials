#!/usr/bin/env python
# coding: utf-8

# ## torch import

# In[1]:


import torch
import numpy as np

# version 체크
print(torch.__version__)


# ## 기본 텐서 생성

# In[2]:


# 샘플 numpy array 생성
arr = np.arange(0, 5)
print(arr)


# ### `torch.from_numpy()`
# 
# - numpy array 로부터 생성. **sharing** 하므로 numpy array의 요소가 변경되면 tensor로 같이 변경됩니다

# In[3]:


t1 = torch.from_numpy(arr)
print(t1) # 출력
print(t1.dtype)  # dtype은 데이터 타입
print(t1.type()) # type()은 텐서의 타입
print(type(t1))  # t1 변수 자체의 타입


# ### `torch.as_tensor()`
# 
# - numpy array 로부터 생성. **sharing** 하므로 numpy array의 요소가 변경되면 tensor로 같이 변경됩니다

# In[4]:


t2 = torch.as_tensor(arr)
print(t2) # 출력
print(t2.dtype)  # dtype은 데이터 타입
print(t2.type()) # type()은 텐서의 타입
print(type(t2))  # t2 변수 자체의 타입


# ### `torch.tensor()` 
# 
# - numpy array 로부터 생성. **copying** 하므로 numpy array의 요소가 변경에 영향을 받지 않습니다.

# In[17]:


t3 = torch.tensor(arr)
print(t3) # 출력
print(t3.dtype)  # dtype은 데이터 타입
print(t3.type()) # type()은 텐서의 타입
print(type(t3))  # t3 변수 자체의 타입


# ## Zeros, Ones

# ### `torch.zeros()`
# 
# - 0으로 채워진 tensor를 생성합니다.
# - dtype을 직접 지정하는 것이 바람직합니다.

# In[21]:


zeros = torch.zeros(3, 5, dtype=torch.int32)
print(zeros)
print(zeros.dtype)
print(zeros.type())


# ### `torch.ones()`
# 
# - 1로 채워진 tensor를 생성합니다.
# - 역시 dtype을 직접 지정하는 것이 바람직합니다.

# In[22]:


ones = torch.zeros(2, 3, dtype=torch.int64)
print(ones)
print(ones.dtype)
print(ones.type())


# ### Tensors from ranges
# <a href='https://pytorch.org/docs/stable/torch.html#torch.arange'><strong><tt>torch.arange(start,end,step)</tt></strong></a><br>
# <a href='https://pytorch.org/docs/stable/torch.html#torch.linspace'><strong><tt>torch.linspace(start,end,steps)</tt></strong></a><br>
# Note that with <tt>.arange()</tt>, <tt>end</tt> is exclusive, while with <tt>linspace()</tt>, <tt>end</tt> is inclusive.

# ## 범위로 생성

# ### `torch.arange(start, end, step)`
# 
# - 지정된 범위로 tensor를 생성합니다.

# In[34]:


# end만 지정
a = torch.arange(5)
print(a)
# start, end 지정
a = torch.arange(2, 6)
print(a)
# start, end, step 모두 지정
a = torch.arange(1, 10, 2)
print(a)


# ### torch.linspace(start, end, steps)
# 
# - start부터 end까지 동일 간격으로 생성합니다. steps 지정시 steps 갯수만큼 생성합니다. (미지정시 100개 생성)

# In[44]:


# start, stop 지정 ()
b = torch.linspace(2, 10)
print(b)
print(b.size(0))
print('==='*20)
# start, stop, step 모두 지정
b = torch.linspace(2, 10, 5)
print(b)


# ## tensor의 타입 변경: type()
# 
# - tensor의 dtype을 변경하기 위해서는 type() 함수를 사용합니다. type()함수의 인자로 변경할 tensor의 타입을 지정합니다.

# In[52]:


aa = torch.arange(10, dtype=torch.int32)
print(aa)
print(aa.type())

print('==='*10)
# tensor의 타입 변경
bb = aa.type(torch.int64)
print(bb)
print(bb.type())


# ## 랜덤 tensor 생성
# 
# - `torch.rand()`: [0, 1) 분포 안에서 랜덤한 tensor를 생성합니다.
# - `torch.randn()`: **standard normal** 분포 안에서 랜덤한 tensor를 생성합니다.
# - `torch.randint()`: 정수로 채워진 랜덤한 tensor를 생성합니다.

# In[53]:


# random 생성 범위: 0 ~ 1 
rd1 = torch.rand(2, 3)
print(rd1)


# In[55]:


# random 생성 범위: standard normal
rd2 = torch.randn(2, 3)
print(rd2)


# In[63]:


# randint 생성시 low, high, size를 지정한 경우
rd3 = torch.randint(low=1, high=10, size=(2, 3))
print(rd3)


# `torch.manual_seed()`: 난수 생성시 시드의 고정

# In[70]:


# manual_seed를 고정시 고정한 cell의 난수 생성은 매번 동일한 값을 생성
torch.manual_seed(0)
rd4 = torch.randint(low=1, high=100, size=(2, 3))
print(rd4)


# ## like로 tensor 생성
# 
# - 텐서 생성 함수에서 `_like()`가 뒤에 붙는 함수를 종종 볼 수 있습니다.
# - `_like()`가 붙은 이름의 함수는 `_like()` 안에 넣어주는 tensor의 shape와 동일한 tensor를 생성합니다.
# 
# **_like() 함수 목록**
# - `torch.rand_like()`
# - `torch.randn_like()`
# - `torch.randint_like()`
# - `torch.ones_like()`
# - `torch.zeros_like()`

# In[87]:


x = torch.tensor([[1, 3, 5], 
                  [7, 9, 11]], dtype=torch.float32)
print(x)
print(x.type())


# In[101]:


# [0, 1)
like1 = torch.rand_like(x)
print(like1)
print(like1.type())


# In[100]:


# standard normal
like2 = torch.randn_like(x)
print(like2)
print(like2.type())


# In[99]:


# int range
like3 = torch.randint_like(x, low=1, high=100)
print(like3)
print(like3.type())


# In[98]:


# zeros
like4 = torch.zeros_like(x)
print(like4)
print(like4.type())


# In[97]:


# ones
like5 = torch.ones_like(x)
print(like5)
print(like5.type())


# ## tensor의 shape 확인 및 변경

# In[110]:


x = torch.tensor([[1, 3, 5], 
                  [7, 9, 11]], dtype=torch.float32)
print(x)


# ### shape 확인

# In[107]:


print(x.shape)
print(x.shape[0])
print(x.shape[1])


# In[108]:


print(x.size())
print(x.size(0))
print(x.size(1))


# ### shape 변경

# `view()`와 `reshape()` 모두 사용가능합니다.

# In[111]:


x = torch.tensor([[1, 3, 5], 
                  [7, 9, 11]], dtype=torch.float32)
print(x)
print(x.shape)


# In[116]:


print(x)
# view()로 shape 변경
print(x.view(3, 2))


# In[117]:


# view
x.view(-1, 1)


# In[120]:


# reshape
x.reshape(-1, 1)


# In[119]:


# view
x.view(3, -1)


# In[121]:


# reshape
x.reshape(3, -1)

