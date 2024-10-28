import torch
import torch.nn.functional as F
from modules import Base, operations
from modules.transform import Transform


class StreamLinear(Base.BaseLayer):
    "Linear layer for stream tensor (multiplication only, no summing)"

    def __init__(self, in_feature, out_feature, seq_len):
        """
        StreamLinear : Linear layer for stream tensor (multiplication only, no summing)
        input (batch_size, in_feature, seq_len)
        output (batch_size, out_feature, in_feature, seq_len)

        Parameters
        ----------
        in_feature : int
            in feature
        out_feature : int
            out feature
        seq_len : int
            sequence length
        """
        super(StreamLinear, self).__init__(seq_len)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = torch.nn.Parameter(torch.Tensor(out_feature, in_feature))
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)

    def generate_Sparams(self):
        "genreate the params for stream deduction"
        self.Sweight = self.trans.f2s(self.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return (x * self.weight + x + self.weight.unsqueeze(0) + 1) / 2 - 1

    def Sforward(self, stream: torch.Tensor) -> torch.Tensor:
        assert stream.size(-1) == self.seq_len, "seq_len not aligned"
        return torch.logical_and(stream.unsqueeze(1), self.Sweight)


class StreamConv(Base.BaseLayer):
    "Convolution layer for stream tensor"

    def __init__(
        self, seq_len, in_channels, out_channels, kernel=3, stride=1, padding=0
    ):
        """StreamConv: Convolution layer for stream tensor

        input (batch_size, in_channels, height, width, seq_len)
        input (batch_size, out_channels, height, width, seq_len)
        output consists of integers ranging from [0, in_feature]

        Parameters
        ----------
        seq_len : int
            sequence length
        in_channels : int
            number of in channels
        out_channels : int
            number of out channels
        kernel_size : int, optional
            size of kernel, by default 3
        stride : int, optional
            stride, by default 1
        padding : int, optional
            padding size, by default 0
        """
        super(StreamConv, self).__init__(seq_len)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel, kernel)
        )
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)

    def generate_Sparams(self):
        "genreate the params for stream deduction"
        self.Sweight = self.trans.f2s(self.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding,) * 4, mode="constant", value=-1)
        return F.conv2d(x, self.weight, None, self.stride)

    def Sforward(self, stream: torch.Tensor):
        assert stream.size(-1) == self.seq_len, "seq_len not aligned"
        stream = F.pad(
            stream.swapaxes(-1, -3), (self.padding,) * 4, "constant", 0
        ).swapaxes(-1, -3)
        batch_size = stream.size(0)
        input_size = stream.size(-2)
        output_size = (input_size - self.kernel) // self.stride + 1
        output = torch.empty(
            batch_size, self.out_channels, output_size, output_size, self.seq_len
        )
        expanded_weight = self.Sweight.view(self.out_channels, -1, self.seq_len)
        for i in range(output_size):
            for j in range(output_size):
                part_stream = stream[
                    ...,
                    i * self.stride : i * self.stride + self.kernel,
                    j * self.stride : j * self.stride + self.kernel,
                    :,
                ].reshape(batch_size, -1, self.seq_len)

                output[..., i, j, :] = operations.matmul(part_stream, expanded_weight)

        return output


class BTanh(Base.BaseLayer):
    def __init__(self, seq_len):
        super().__init__(seq_len)

    def generate_Sparams(self):
        return

    def Sforward(self, inputs: torch.Tensor):
        return operations.tanh(inputs.to(int))

    def forward(self, x: torch.Tensor):
        return ((x + 1) ** 2 * (2 - x)) / 2 - 1


class Majority_k(Base.BaseLayer):
    def __init__(self, in_features, k, seq_len):
        super().__init__(seq_len)
        self.in_features = in_features
        self.k = k
        assert (k < in_features) and (k >= 0)

    def generate_Sparams(self):
        return

    def Sforward(self, inputs: torch.Tensor, k=None) -> torch.Tensor:
        """apply Majority_k on input stream

        Parameters
        ----------
        inputs : torch.Tensor
            (batch_size, in_features, seq_len), each (in_features, seq_len) in batch_size is streams of a probability sequence
        k : int, optional
            majority threshold, Majority_0 == any, Majority_(n-1) == all

        Returns
        -------
        torch.Tensor
            (batch_size, seq_len), each (seq_len,) in batch_size is the result stream of Majority_k of input (in_features, seq_len)
        """
        if k is None:
            k = self.k

        inputs = inputs.sum(dim=-2)

        return inputs > k

    def rawforward(self, inputs: torch.Tensor, k=None) -> torch.Tensor:
        """probability calculation of Majority_k, return 1-cdf_Poisson_binomial(k)

        Parameters
        ----------
        inputs : torch.Tensor
            (batch_size, in_features), each row is a probability sequence in range [0,1]
        k : int, optional
            majority threshold, Majority_0 == any, Majority_(n-1) == all

        Returns
        -------
        torch.Tensor
            (batch_size,)
        """
        if k is None:
            k = self.k
        assert self.in_features == inputs.shape[1]
        assert k >= 0 and k < self.in_features
        pmf = torch.zeros(inputs.shape[0], k + 1)
        pmf[:, 0] = 1

        for i in range(self.in_features):
            p = inputs[:, i]
            pmf[:, 1:] = (pmf[:, 1:].T * (1 - p) + pmf[:, :-1].T * p).T
            pmf[:, 0] *= 1 - p

        return 1 - pmf.sum(dim=-1)

    def forward(self, inputs: torch.Tensor, k=None) -> torch.Tensor:
        """float calculation of Majority_k

        Parameters
        ----------
        inputs : torch.Tensor
            (batch_size, in_features), each row is a float sequence in range [-1,1]
        k : int, optional
            majority threshold, Majority_0 == any, Majority_(n-1) == all

        Returns
        -------
        torch.Tensor
            (batch_size,)
        """
        inputs = (inputs + 1) / 2
        inputs = self.rawforward(inputs, k)
        return 2 * inputs - 1
