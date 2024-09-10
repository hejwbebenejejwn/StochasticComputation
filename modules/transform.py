import torch


class Transform:
    """
    Transformer that transform float Tensor to Stream and the other way round  
    """
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def f2s(self, float_tensor: torch.Tensor) -> torch.Tensor:
        """transform float to stream

        Parameters
        ----------
        float_tensor : torch.Tensor
            tensor of shape (...)

        Returns
        -------
        torch.Tensor
            tensor stream of shape (..., seq_len)
        """
        dims = len(float_tensor.shape)
        float_tensor = (float_tensor + 1) / 2
        float_tensor = float_tensor.unsqueeze(-1)
        float_tensor = float_tensor.expand(*(-1,) * dims, self.seq_len)
        return torch.bernoulli(float_tensor)

    def s2f(self, stream_tensor: torch.Tensor) -> torch.Tensor:
        """transorm stream to float, last dim of input tensor should be seq_len

        Parameters
        ----------
        stream_tensor : torch.Tensor
            tensor of shape (..., seq_len)

        Returns
        -------
        torch.Tensor
            tensor of shape (...)
        """
        assert (
            self.seq_len == stream_tensor.shape[-1]
        ), f"wrong stream length, expect {self.seq_len}, got {stream_tensor.shape[-1]}"
        stream_tensor = stream_tensor.sum(-1) / self.seq_len
        return stream_tensor * 2 - 1
