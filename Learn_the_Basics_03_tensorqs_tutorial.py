#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# [Learn the Basics](intro.html) ||
# [Quickstart](quickstart_tutorial.html) ||
# **Tensors** ||
# [Datasets & DataLoaders](data_tutorial.html) ||
# [Transforms](transforms_tutorial.html) ||
# [Build Model](buildmodel_tutorial.html) ||
# [Autograd](autogradqs_tutorial.html) ||
# [Optimization](optimization_tutorial.html) ||
# [Save & Load Model](saveloadrun_tutorial.html)
# 
# # Tensors
# 
# Tensors are a specialized data structure that are very similar to arrays and matrices.
# In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
# 
# Tensors are similar to [NumPy’s](https://numpy.org/) ndarrays, except that tensors can run on GPUs or other hardware accelerators. In fact, tensors and
# NumPy arrays can often share the same underlying memory, eliminating the need to copy data (see `bridge-to-np-label`). Tensors
# are also optimized for automatic differentiation (we'll see more about that later in the [Autograd](autogradqs_tutorial.html)_
# section). If you’re familiar with ndarrays, you’ll be right at home with the Tensor API. If not, follow along!
# 

# In[ ]:


import torch
import numpy as np


# ## Initializing a Tensor
# 
# Tensors can be initialized in various ways. Take a look at the following examples:
# 
# **Directly from data**
# 
# Tensors can be created directly from data. The data type is automatically inferred.
# 
# 

# In[ ]:


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)


# **From a NumPy array**
# 
# Tensors can be created from NumPy arrays (and vice versa - see `bridge-to-np-label`).
# 
# 

# In[ ]:


np_array = np.array(data)
x_np = torch.from_numpy(np_array)


# **From another tensor:**
# 
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
# 
# 

# In[ ]:


x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# **With random or constant values:**
# 
# ``shape`` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
# 
# 

# In[ ]:


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# --------------
# 
# 
# 

# ## Attributes of a Tensor
# 
# Tensor attributes describe their shape, datatype, and the device on which they are stored.
# 
# 

# In[ ]:


tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# --------------
# 
# 
# 

# ## Operations on Tensors
# 
# Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing,
# indexing, slicing), sampling and more are
# comprehensively described [here](https://pytorch.org/docs/stable/torch.html)_.
# 
# Each of these operations can be run on the GPU (at typically higher speeds than on a
# CPU). If you’re using Colab, allocate a GPU by going to Runtime > Change runtime type > GPU.
# 
# By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using
# ``.to`` method (after checking for GPU availability). Keep in mind that copying large tensors
# across devices can be expensive in terms of time and memory!
# 
# 

# In[ ]:


# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")


# Try out some of the operations from the list.
# If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.
# 
# 
# 

# **Standard numpy-like indexing and slicing:**
# 
# 

# In[ ]:


tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)


# **Joining tensors** You can use ``torch.cat`` to concatenate a sequence of tensors along a given dimension.
# See also [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)_,
# another tensor joining op that is subtly different from ``torch.cat``.
# 
# 

# In[ ]:


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


# **Arithmetic operations**
# 
# 

# In[ ]:


# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


# **Single-element tensors** If you have a one-element tensor, for example by aggregating all
# values of a tensor into one value, you can convert it to a Python
# numerical value using ``item()``:
# 
# 

# In[ ]:


agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# **In-place operations**
# Operations that store the result into the operand are called in-place. They are denoted by a ``_`` suffix.
# For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.
# 
# 

# In[ ]:


print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


# <div class="alert alert-info"><h4>Note</h4><p>In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
#      of history. Hence, their use is discouraged.</p></div>
# 
# 

# --------------
# 
# 
# 

# 
# ## Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory
# locations, and changing one will change	the other.
# 
# 

# ### Tensor to NumPy array
# 
# 

# In[ ]:


t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


# A change in the tensor reflects in the NumPy array.
# 
# 

# In[ ]:


t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


# ### NumPy array to Tensor
# 
# 

# In[ ]:


n = np.ones(5)
t = torch.from_numpy(n)


# Changes in the NumPy array reflects in the tensor.
# 
# 

# In[ ]:


np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

