import torch
import numpy as np
from modules.transform import Transform


def mul(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """multiply two tensors of stream, tensor2.shape should be part of tensor1.shape

    Parameters
    ----------
    tensor1 : torch.Tensor
        tensor of shape (..., seq_len)
    tensor2 : torch.Tensor
        tensor of shape (..., seq_len)

    Returns
    -------
    torch.Tensor
        tensor of same shape as tensor1
        if the element at (...) is (...) of tensor1 * (...) of tensor2 (in float sense)
    """
    return torch.logical_not(torch.logical_xor(tensor1, tensor2))


class APCounter:
    def __init__(self, in_features=16, num_au_layers=1):
        self.num_au_layers = num_au_layers
        self.in_features = in_features

    def approx_unit(self, x: torch.tensor) -> torch.tensor:
        seq_len = x.shape[-1]
        half = int(seq_len / 2)

        idx1, idx2, idx3, idx4 = [torch.arange(i, x.shape[-2], 4) for i in range(4)]
        grp1, grp2, grp3, grp4 = (
            x[..., idx1, :],
            x[..., idx2, :],
            x[..., idx3, :],
            x[..., idx4, :],
        )

        out1 = torch.concat(
            [
                torch.logical_or(grp1[..., :half], grp2[..., :half]),
                torch.logical_and(grp3[..., :half], grp4[..., :half]),
            ],
            dim=-2,
        )
        out2 = torch.concat(
            [
                torch.logical_and(grp1[..., half:], grp2[..., half:]),
                torch.logical_or(grp3[..., half:], grp4[..., half:]),
            ],
            dim=-2,
        )
        return torch.concat([out1, out2], dim=-1)

    def counter(self, x):
        return torch.sum(x, dim=-2, dtype=torch.int16)

    def __call__(self, x):
        count = 0
        for i in range(self.num_au_layers):
            num_units = int(x.shape[-2] / 4)
            x, res = x[..., 0 : 4 * num_units, :], x[..., 4 * num_units :, :]
            x = self.approx_unit(x)
            count += torch.tensor(2**i, dtype=torch.int16) * self.counter(res)
        return (
            torch.tensor(2**self.num_au_layers, dtype=torch.int16) * self.counter(x)
            + count
        )


def matmul(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    matrix multiplication

    Parameters
    ----------
    tensor1 : torch.Tensor
        tensor of shape (a, b, seq_len)
    tensor2 : torch.Tensor
        tensor of shape (c, b, seq_len)

    Returns
    -------
    torch.Tensor
        tensor of shape (a, c, seq_len)
    """
    a, b, seq_len = tensor1.shape
    c = tensor2.size(0)
    trans = Transform(seq_len)
    apc = APCounter(b)
    output = torch.empty(a, c, seq_len)
    for i in range(c):
        count = apc(mul(tensor1, tensor2[i]))
        output[:, i, :] = trans.f2s((count.sum(dim=-1) / seq_len * 2 - b) / b)
    return output
