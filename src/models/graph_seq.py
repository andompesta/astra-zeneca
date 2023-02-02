import torch
from torch import nn, Tensor, LongTensor

from src.models.graph_encoder import GraphEncoder
from src.models.seq_decoder import DecoderRNN, AttnDecoderRNN


class GraphSeq(torch.nn.Module):
    """
    Simple graph-to-seq architecture that does not use attention.
    Use this for prototyping
    """

    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        pad_idx: int,
        # graph params
        graph_conv_layers: int,

        # rnn decoder parmas
        rnn_decoder_layers: int,
        rnn_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(
            vocab_size,
            self.emb_dim,
            padding_idx=pad_idx,
        )
        self.graph_enc = GraphEncoder(
            self.emb_dim,
            graph_conv_layers,
        )
        self.seq_dec = DecoderRNN(
            self.emb_dim,
            vocab_size,
            rnn_decoder_layers,
            rnn_dropout,
        )

        self.out = nn.Linear(emb_dim, vocab_size)

        self.reset_parameter()

    def forward(
        self,
        node_idx: LongTensor,
        src_idx: LongTensor,
        fw_edge_idx: LongTensor,
        bw_edge_id: LongTensor,
        batch: LongTensor,
    ) -> Tensor:
        node_emb = self.emb(node_idx)
        src_seq = self.emb(src_idx)

        graph_emb, node_emb = self.graph_enc(
            node_emb,
            fw_edge_idx,
            bw_edge_id,
            batch,
        )

        context = self.seq_dec.init_hidden(graph_emb)
        trg_pred, _ = self.seq_dec(src_seq, context)
        trg_pred = self.out(trg_pred)
        return trg_pred

    def reset_parameter(self):
        torch.nn.init.xavier_normal_(self.emb.weight.data)
        self.graph_enc.reset_parameter()
        self.seq_dec.reset_parameter()


class GraphSeqAttn(torch.nn.Module):
    """
    Graph-to-seq architecture with attention.
    """

    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        pad_idx: int,
        # graph params
        graph_conv_layers: int,

        # rnn decoder parmas
        rnn_decoder_layers: int,
        rnn_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(
            vocab_size,
            self.emb_dim,
            padding_idx=pad_idx,
        )
        self.graph_enc = GraphEncoder(
            self.emb_dim,
            graph_conv_layers,
        )
        self.seq_dec = AttnDecoderRNN(
            self.emb_dim,
            vocab_size,
            rnn_decoder_layers,
            rnn_dropout,
        )
        self.out = nn.Linear(emb_dim, vocab_size)

        self.reset_parameter()

    def forward(
        self,
        node_idx: LongTensor,
        src_idx: LongTensor,
        fw_edge_idx: LongTensor,
        bw_edge_id: LongTensor,
        batch: LongTensor,
    ) -> Tensor:
        node_emb = self.emb(node_idx)
        src_seq = self.emb(src_idx)

        graph_emb, node_emb = self.graph_enc(
            node_emb,
            fw_edge_idx,
            bw_edge_id,
            batch,
        )

        hidden = self.seq_dec.init_hidden(graph_emb)
        trg_pred, _ = self.seq_dec(
            src_seq,
            hidden,
            node_emb,
            batch,
        )

        trg_pred = self.out(trg_pred)
        return trg_pred

    def reset_parameter(self):
        torch.nn.init.xavier_normal_(self.emb.weight.data)
        torch.nn.init.xavier_normal_(self.out.weight.data)
        torch.nn.init.zeros_(self.out.bias.data)

        self.graph_enc.reset_parameter()
        self.seq_dec.reset_parameter()
