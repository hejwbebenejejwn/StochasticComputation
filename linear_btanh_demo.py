from modules.transform import Transform
from modules.layers import BTanh,StreamLinear
from modules.operations import APCounter
import torch

SEQ_LEN = 1000000
trans=Transform(SEQ_LEN)
linear=StreamLinear(28*28,128,SEQ_LEN)
btanh=BTanh(SEQ_LEN,28*28)
r=2300

a=2*torch.rand(1,28*28)-1
sa=trans.f2s(a)
a=linear(a)
a=btanh(a)
linear.generate_Sparams()
sa=linear.Sforward(sa)
print('linear')
sa=btanh.Sforward(sa,r)

