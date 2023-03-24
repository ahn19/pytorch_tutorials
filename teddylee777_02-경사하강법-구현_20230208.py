#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
import matplotlib.pyplot as plt


# ## 샘플 데이터셋 생성

# In[42]:


x = torch.tensor(np.arange(10))
y = 2*x + 1
y


# In[62]:


plt.title('y = 2x + b', fontsize=15)
plt.scatter(x, y)
plt.show()


# In[47]:


# random w, b 생성
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
w, b


# In[48]:


# Hypothesis Function 정의
y_hat = w*x + b


# In[49]:


# Mean Squared Error(MSE) 오차 정의
loss = ((y_hat - y)**2).mean()


# In[50]:


# BackPropagation (Gradient 계산)
loss.backward()


# In[56]:


# 결과 출력
print(f'w gradient: {w.grad.item():.2f}, b gradient: {b.grad.item():.2f}')


# ## w, b 의 직접 계산한 Gradient와 비교

# In[58]:

# https://towardsdatascience.com/implementing-linear-regression-with-gradient-descent-from-scratch-f6d088ec1219
# 에 유도공식이 있는데, 잘 구현된 것임을 알았음
w_grad = (2*(y_hat - y)*x).mean().item()
b_grad = (2*(y_hat - y)).mean().item()


# In[59]:


print(f'w gradient: {w_grad:.2f}, b gradient: {b_grad:.2f}')


# ## Gradient 계산 미적용

# In[60]:


y_hat = w*x + b
print(y_hat.requires_grad)

with torch.no_grad():
    y_hat = w*x + b
    
print(y_hat.requires_grad)

