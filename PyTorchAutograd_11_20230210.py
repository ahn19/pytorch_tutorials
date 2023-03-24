
# https://teamdable.github.io/techblog/PyTorch-Autograd

import torch

def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name, None)})')
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

class MyPow(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input_1, input_2):
    ctx.save_for_backward(input_1, input_2)
    result = input_1 ** input_2
    return result

  @staticmethod
  def backward(ctx, grad_output):
    input_1, input_2 = ctx.saved_tensors
    grad_input_1 = grad_output * input_2 * input_1 ** (input_2 - 1)
    grad_input_2 = grad_output * input_1 ** input_2 * torch.log(input_1)
    print('input_1', input_1)
    print('input_2', input_2)
    print('grad_output', grad_output)
    print('grad_input_1', grad_input_1)
    print('grad_input_2', grad_input_2)
    return grad_input_1, grad_input_2

myPow = MyPow.apply

q = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(5.0, requires_grad=True)
y = myPow(x, q)
z = torch.log(y)

print('q', get_tensor_info(q))
print('x', get_tensor_info(x))
print('y', get_tensor_info(y))
print('z', get_tensor_info(z))

z.backward()

print('q_after_backward', get_tensor_info(q))
print('x_after_backward', get_tensor_info(x))
print('y_after_backward', get_tensor_info(y))
print('z_after_backward', get_tensor_info(z))

print('q.grad', q.grad)
print('x.grad', x.grad)
print('y.grad', y.grad)
print('z.grad', z.grad)

a1 = 1

# q requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(None) tensor(tensor(3., requires_grad=True))
# x requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(None) tensor(tensor(5., requires_grad=True))
# y requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<torch.autograd.function.MyPowBackward object at 0x7f96e247e4a8>) grad(None) tensor(tensor(125., grad_fn=<MyPowBackward>))
# z requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<LogBackward object at 0x7f970c1db780>) grad(None) tensor(tensor(4.8283, grad_fn=<LogBackward>))
# input_1 tensor(5., requires_grad=True)
# input_2 tensor(3., requires_grad=True)
# grad_output tensor(0.0080)
# grad_input_1 tensor(0.6000)
# grad_input_2 tensor(1.6094)
# q_after_backward requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(1.6094379425048828) tensor(tensor(3., requires_grad=True))
# x_after_backward requires_grad(True) is_leaf(True) retains_grad(None) grad_fn(None) grad(0.6000000238418579) tensor(tensor(5., requires_grad=True))
# y_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<torch.autograd.function.MyPowBackward object at 0x7f96e247e4a8>) grad(None) tensor(tensor(125., grad_fn=<MyPowBackward>))
# z_after_backward requires_grad(True) is_leaf(False) retains_grad(None) grad_fn(<LogBackward object at 0x7f970c1dba58>) grad(None) tensor(tensor(4.8283, grad_fn=<LogBackward>))










































