import torch
import torch.nn as nn
from modules import transform
from abc import ABC, abstractmethod


class BaseLayer(nn.Module, ABC):
    "Base class for Stream Layer"

    def __init__(self, seq_len):
        super(BaseLayer, self).__init__()
        self.seq_len = seq_len
        self.trans = transform.Transform(seq_len)

    @abstractmethod
    def generate_Sparams(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def Sforward(self, stream: torch.Tensor):
        pass


class BaseModel(nn.Module, ABC):
    "Base class for Stream Model"

    def __init__(self, seq_len):
        super(BaseModel, self).__init__()
        self.seq_len = seq_len

    def read_params(self, model: nn.Module):
        """read the params from a source model

        Parameters
        ----------
        model : nn.Module
            source model
        """
        dict1, dict2 = self.state_dict(), model.state_dict()
        dict1.update(dict2)
        self.load_state_dict(dict1, strict=False)

    def generate_Sparams(self):
        for layer in self.modules():
            if isinstance(layer, BaseLayer):
                layer.generate_Sparams()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def Sforward(self, stream: torch.Tensor):
        pass
