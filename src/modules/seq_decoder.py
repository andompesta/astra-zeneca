import torch
from torch import nn


class DecoderRNN(nn.Module):

    def __init__(
        self,
        emd_dim: int,
        vocab_size: int,
        layers: int,
        drop_prob: float = 0.5,
    ):
        super(DecoderRNN, self).__init__()
        self.emd_dim = emd_dim
        self.vocab_size = vocab_size
        self.layers = layers

        self.gru = nn.GRU(
            emd_dim,
            emd_dim,
            batch_first=True,
            dropout=drop_prob,
            num_layers=layers,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(emd_dim, vocab_size)

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)
        x = self.out(x)
        return x, hidden

    def init_hidden(self, graph_enc: torch.Tensor):
        return graph_enc.unsqueeze(0).expand(self.layers, -1, -1).contiguous()

    def reset_parameter(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            else:
                raise NotImplementedError(
                    "plain parameters are not supported \t {}".format(name))
