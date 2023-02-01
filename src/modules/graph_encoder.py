
from torch_geometric import nn
from torch_scatter import scatter


class MPNNLayer(nn.MessagePassing):
    # this implement the forward aggregation process only
    # backward embedding is obtained by reversing the adges direction in the graph
    def __init__(
        self,
        emb_dim=4,
        aggr='add',
        node_dim=-2,
        # aggregate forward neitghours
        flow="target_to_source",
    ):
        super().__init__(
            aggr=aggr,
            node_dim=node_dim,
            flow=flow,
        )

        self.emb_dim = emb_dim

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

    def message(self, h_i, h_j):
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
        msg = h_i + h_j
        return msg

    def aggregate(self, inputs, index):
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

    def update(self, aggr_out, h):
        """
        Step (3) Update

        The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`, 
        as well as any optional arguments that were initially passed to 
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        return h + aggr_out