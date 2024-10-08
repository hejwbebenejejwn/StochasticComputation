import torch
import torch.nn.functional as F
from modules import Base, operations
from modules.transform import Transform


class StreamLinear(Base.BaseLayer):
    "Linear layer for stream tensor"

    def __init__(self, in_feature, out_feature, seq_len):
        """
        StreamLinear : Linear layer for stream tensor
        input (batch_size, in_feature, seq_len)
        output (batch_size, out_feature, seq_len)
        output consists of integers ranging from [0, in_feature]

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
        self.apc = operations.APCounter(in_feature)
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)

    def generate_Sparams(self):
        "genreate the params for stream deduction"
        self.Sweight = self.trans.f2s(self.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, None)

    def Sforward(self, stream: torch.Tensor) -> torch.Tensor:
        assert stream.size(-1) == self.seq_len, "seq_len not aligned"
        return operations.matmul(stream, self.Sweight)


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
    def __init__(self, seq_len, in_feature):
        super().__init__(seq_len)
        self.in_feature = in_feature


    def generate_Sparams(self):
        return

    def Sforward(self, inputs: torch.Tensor, r: int):
        s_max = r - 1
        s_half = (r - 1) / 2

        s = torch.full(inputs.shape[:-1], s_half - 0.5, dtype=torch.int16)
        inputs = inputs * 2 - self.in_feature

        out = torch.zeros(inputs.shape, dtype=torch.bool)

        for i in range(self.seq_len):
            s = s + inputs[..., i]
            overflow_idx = s > s_max
            underflow_idx = s < 0
            s[overflow_idx] = s_max
            s[underflow_idx] = 0
            out[..., i] = s > s_half
        return out

    def forward(self, inputs, scale=1):
        return torch.tanh(scale * inputs)
