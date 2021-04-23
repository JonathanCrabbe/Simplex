import torch

a = torch.tensor([1., 1.], requires_grad=True)
b = torch.tensor([2., 2.], requires_grad=True)
Q = 3 * a**2 + 5 * b**3
external_vector = torch.tensor([1., 1.])
Q.backward(gradient=external_vector)
print(a.grad)
print(b.grad)