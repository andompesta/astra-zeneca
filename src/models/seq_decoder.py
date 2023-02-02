import torch
from torch import nn, Tensor


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

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)
        return x, hidden

    def init_hidden(self, graph_enc: torch.Tensor):
        return graph_enc.unsqueeze(0).expand(self.layers, -1, -1).contiguous()
        # return torch.zeros_like(
        #     graph_enc.unsqueeze(0).expand(self.layers, -1, -1)).contiguous()

    def reset_parameter(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
            else:
                raise NotImplementedError(
                    "plain parameters are not supported \t {}".format(name))


class AttnDecoderRNN(nn.Module):

    def __init__(
        self,
        emd_dim: int,
        vocab_size: int,
        layers: int,
        drop_prob: float = 0.5,
    ):
        super(AttnDecoderRNN, self).__init__()
        self.emd_dim = emd_dim
        self.vocab_size = vocab_size
        self.layers = layers

        self.gru = nn.GRU(
            emd_dim * 2,
            emd_dim,
            batch_first=True,
            dropout=drop_prob,
            num_layers=layers,
        )
        self.dropout = nn.Dropout(drop_prob)

        # layer to get the query from tokens
        self.attn_seq = nn.Linear(
            self.emd_dim,
            self.emd_dim,
            bias=False,
        )

        # layer to get the keys from nodes
        self.attn_nodes = nn.Linear(
            self.emd_dim,
            self.emd_dim,
            bias=False,
        )

    def forward(
        self,
        x: Tensor,
        hidden: Tensor,
        encoder_output: Tensor,
        node_batch: Tensor,
    ):
        batch_size = x.size(0)

        # attention
        encoder_output = encoder_output.unsqueeze(0).expand(batch_size, -1,
                                                            -1)  # B, V, H

        # get query
        att_query = self.attn_seq(x)  # B, L, H
        # get keys
        att_key = self.attn_nodes(encoder_output)  # B, V, H

        # Note a token can attend to all nodes, event the one not belonging to its graph
        mask = torch.stack([
            (node_batch != i).float() * -999999. for i in range(batch_size)
        ]).unsqueeze_(1)

        att_probs = att_query.bmm(att_key.transpose(-1, -2))  # B, L, V
        att_probs = torch.nn.functional.softmax(
            att_probs + mask,
            dim=-1,
        )  # B, L, V

        context = att_probs.bmm(encoder_output)  # B, L, H

        x = torch.cat((x, context), dim=-1)  # B, L, H

        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)

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
