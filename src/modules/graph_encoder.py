import torch
from torch import Tensor, nn

from torch_geometric import nn as gnn
from torch_scatter import scatter


class MPNNLayer(gnn.MessagePassing):
    # this implement the forward aggregation process only
    # backward embedding is obtained by reversing the adges direction in the graph
    def __init__(
        self,
        emb_dim: int,
        aggr: str = 'max',
        node_dim: int = -2,
        # aggregate forward neitghours
        flow: str = "target_to_source",
    ):
        super().__init__(
            aggr=aggr,
            node_dim=node_dim,
            flow=flow,
        )

        self.emb_dim = emb_dim

        self.ln_pool = torch.nn.Linear(
            self.emb_dim,
            self.emb_dim,
        )

        self.ln_merge = torch.nn.Linear(
            2 * self.emb_dim,
            emb_dim,
            bias=False,
        )

    def forward(self, h, edge_index):
        """
        The forward pass updates node features `h` via one round of message passing.
        Args:
            h: (n, d) - initial node features
            edge_index: (2, e) - pairs of edges (i, j)

        Returns:
            out: (n, d) - updated node features
        """

        self.node_size = h.size(0)

        out = self.propagate(
            edge_index=edge_index,
            h=h,
        )
        return out

    def message(self, h_j: Tensor) -> Tensor:
        """
        Create messages for each edge based on node embedding.
        Since it is the embedding for the forward pass, i represent the sournce node
        while j represent the destination node for each edge (i, j).

        Destination nodes and source nodes embedding are represented by appending 
        `_i` or `_j` to the variable name.

        Args:
            h_i: (e, d) - source node features
            h_j: (e, d) - destination node features

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        return h_j
        # return torch.nn.functional.relu(self.ln_pool(h_j))

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
    ) -> Tensor:
        """Aggregates the messages from neighboring nodes. In this case 
        the neighours are the destination nodes of each edge.
        The aggregation function applied is defined by `aggr` parameter.

        Args:
            inputs: (e, d) - messages `m_ij` from source to destination nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(
            inputs,
            index,
            dim=self.node_dim,
            # zero-pad missing node embedding
            dim_size=self.node_size,
            reduce=self.aggr,
        )

    def update(
        self,
        aggr_out: Tensor,
        h: Tensor,
    ) -> Tensor:
        """The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features
        """
        return h + aggr_out
        h = self.ln_merge(torch.cat(
            (h, aggr_out),
            dim=-1,
        ))
        h = torch.nn.functional.relu(h)
        return h


class GraphEncoder(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        layers: int,
        aggr: str = "sum",
        **kwargs,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        self.convs = nn.ModuleList([
            MPNNLayer(
                self.emb_dim,
                aggr=aggr,
            ) for _ in range(layers)
        ])

        self.ln_edge_merge = nn.Linear(
            2 * self.emb_dim,
            self.emb_dim,
        )

        self.graph_pool = gnn.pool.global_max_pool

    def forward(
        self,
        h,
        fw_edge_index,
        bw_edge_index,
        batch,
    ) -> tuple[Tensor, Tensor]:
        fw_h = h
        for conv in self.convs:
            fw_h = conv(fw_h, fw_edge_index)

        bw_h = h
        for conv in self.convs:
            bw_h = conv(bw_h, bw_edge_index)

        # node embedding used for attention
        h = torch.cat((fw_h, bw_h), dim=-1)
        h = self.ln_edge_merge(h)

        # compute graph embedding by max_pooling
        g_h = self.graph_pool(h, batch)

        return g_h, h

    def reset_parameter(self):
        for name, param in self.named_parameters():
            if name.endswith("weight"):
                # relu and linear layers uses almost the same gain
                nn.init.kaiming_normal_(param.data, nonlinearity="relu")
            elif name.endswith("bias"):
                nn.init.zeros_(param.data)
            else:
                raise NotImplementedError("plain parameters are not supported \t {}".format(name))