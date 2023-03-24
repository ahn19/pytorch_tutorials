
# https://teamdable.github.io/techblog/PyTorch-Autograd

import torch

def get_tensor_info(tensor):
  info = []
  #for name in ['requires_grad', 'is_leaf', 'grad_fn', 'grad']:
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    a1 = getattr(tensor, name)
    info.append(f'{name}({getattr(tensor, name)})')
    # getattr(): object의 속성(attribute) 값을 확인하는 함수
    # https://technote.kr/249
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

x = torch.tensor(5.0)
y = x ** 3
z = torch.log(y)

print('x', get_tensor_info(x))
print('y', get_tensor_info(y))
print('z', get_tensor_info(z))

a1 = 1

# x requires_grad(False) is_leaf(True) grad_fn(None) grad(None) tensor(tensor(5.))
# y requires_grad(False) is_leaf(True) grad_fn(None) grad(None) tensor(tensor(125.))
# z requires_grad(False) is_leaf(True) grad_fn(None) grad(None) tensor(tensor(4.8283))











































