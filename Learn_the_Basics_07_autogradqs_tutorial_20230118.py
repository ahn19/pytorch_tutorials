
# # Automatic Differentiation with ``torch.autograd``

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# ## Tensors, Functions and Computational graph
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
# https://gaussian37.github.io/dl-pytorch-gradient/
# grad_fn에는 텐서가 어떤 연산을 하였는 지 연산 정보를 담고 있고, 이 정보는 역전파 과정에 사용될 예정입니다.
# 각 텐서에서 나중에 이루어지는 역전파를 위해 기록되는 grad_fn에는 각각 AddBackward0,
# SubBackward0, MulBackward0, DivBackward0와 같이 저장되어 있는 것을 확인할 수 있습니다.

# ## Computing Gradients
loss.backward()
print(w.grad)
print(b.grad)

# ## Disabling Gradient Tracking
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
# 동일한 결과를 얻는 또 다른 방법은 텐서에서 detach() 메서드를 사용하는 것입니다 (Another way to achieve the same result is to use the detach() method on the tensor).
# 그래디언트 추적을 비활성화하려는 이유는 다음과 같습니다.
# 1. 신경망의 일부 매개변수를 고정 매개변수로 표시합니다. 이것은 사전 훈련된 네트워크를 미세 조정하는 매우 일반적인 시나리오입니다.
# 2. 그래디언트를 추적하지 않는 텐서에 대한 계산이 더 효율적이기 때문에 순방향 패스만 수행할 때 계산 속도를 높이려면.
#https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

# ## More on Computational Graphs
# ## Optional Reading: Tensor Gradients and Jacobian Products
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

# ### Further Reading




























