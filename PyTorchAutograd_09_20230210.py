
# https://teamdable.github.io/techblog/PyTorch-Autograd

import torch

def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name, None)})')
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

x = torch.tensor(5.0, requires_grad=True)
y = x ** 3
w = x ** 2
z = torch.log(y) + torch.sqrt(w)

print('x', get_tensor_info(x))
print('y', get_tensor_info(y))
print('w', get_tensor_info(w))
print('z', get_tensor_info(z))

z.backward()

print('x_after_backward', get_tensor_info(x))
print('y_after_backward', get_tensor_info(y))
print('w_after_backward', get_tensor_info(w))
print('z_after_backward', get_tensor_info(z))

print('x.grad', x.grad)
print('y.grad', y.grad)
print('w.grad', w.grad)
print('z.grad', z.grad)

a1 = 1

# x requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(None) tensor(tensor(5., requires_grad=True))
# y requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<PowBackward0 object at 0x7f2ed092ca20>) grad(None) tensor(tensor(125., grad_fn=<PowBackward0>))
# w requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<PowBackward0 object at 0x7f2ed092ca20>) grad(None) tensor(tensor(25., grad_fn=<PowBackward0>))
# z requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<AddBackward0 object at 0x7f2ed092ca20>) grad(None) tensor(tensor(9.8283, grad_fn=<AddBackward0>))
# x_after_backward requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(1.600000023841858) tensor(tensor(5., requires_grad=True))
# y_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<PowBackward0 object at 0x7f2ed092ca20>) grad(None) tensor(tensor(125., grad_fn=<PowBackward0>))
# w_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<PowBackward0 object at 0x7f2ed092ca20>) grad(None) tensor(tensor(25., grad_fn=<PowBackward0>))
# z_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<AddBackward0 object at 0x7f2ed092ca20>) grad(None) tensor(tensor(9.8283, grad_fn=<AddBackward0>))







































