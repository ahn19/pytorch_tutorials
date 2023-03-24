
# https://teamdable.github.io/techblog/PyTorch-Autograd

import torch

def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    a1 = getattr(tensor, name)
    a2 = getattr(tensor, name, None)
    info.append(f'{name}({getattr(tensor, name, None)})')
    # getattr(): object의 속성(attribute) 값을 확인하는 함수
    # https://technote.kr/249
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

x = torch.tensor(5.0, requires_grad=True)
y = x ** 3
z = torch.log(y)

print('x', get_tensor_info(x))
print('y', get_tensor_info(y))
print('z', get_tensor_info(z))

z.backward()

print('x_after_backward', get_tensor_info(x))
print('y_after_backward', get_tensor_info(y))
print('z_after_backward', get_tensor_info(z))

print('x.grad', x.grad)
print('y.grad', y.grad)
print('z.grad', z.grad)

a1 = 1

# x requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(None) tensor(tensor(5., requires_grad=True))
# y requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<PowBackward0 object at 0x7f7822fe19e8>) grad(None) tensor(tensor(125., grad_fn=<PowBackward0>))
# z requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<LogBackward object at 0x7f7822fe19e8>) grad(None) tensor(tensor(4.8283, grad_fn=<LogBackward>))
# x_after_backward requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(0.6000000238418579) tensor(tensor(5., requires_grad=True))
# y_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<PowBackward0 object at 0x7f7822fe19e8>) grad(None) tensor(tensor(125., grad_fn=<PowBackward0>))
# z_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<LogBackward object at 0x7f7822fe19e8>) grad(None) tensor(tensor(4.8283, grad_fn=<LogBackward>))












































