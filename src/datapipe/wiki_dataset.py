import torch
import pickle
import networkx as nx
from pathlib import Path

from torch_geometric.data import Data, InMemoryDataset
from src.utils.common import (
    EOS,
    PAD,
    SOS,
    OOV,
    SELECT,
    AND,
    WHERE,
)

MAX_SEQ_LEN = 17


class WikiDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        file_name: str,
        entity_to_id_path: str,
        transform=None,
        pre_transform=None,
    ):
        self.file_name = file_name
        with open(entity_to_id_path, "rb") as f:
            entity_2_id = pickle.load(f)
        self.entity_2_id = entity_2_id

        super().__init__(
            root,
            transform,
            pre_transform,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            f'{self.file_name}_graphs.bin',
            f'{self.file_name}_query_tokens_ids.bin',
        ]

    @property
    def processed_file_names(self):
        return [f'{self.file_name}.pt']

    def download(self):
        pass

    def process(self):
        graphs_path = Path(
            self.raw_dir
        ).joinpath(self.raw_file_names[0])
        with open(graphs_path.as_posix(), "rb") as f:
            graphs = pickle.load(f)

        tokens_path = Path(
            self.raw_dir
        ).joinpath(self.raw_file_names[1])
        with open(tokens_path.as_posix(), "rb") as f:
            tokens_ids = pickle.load(f)

        assert len(graphs) == len(tokens_ids)
        data_list = []

        for graph, token_ids in zip(graphs, tokens_ids):
            graph_num = nx.convert_node_labels_to_integers(graph)
            edges_idx = list(graph_num.out_edges)
            fw_edge_idx = torch.tensor(edges_idx, dtype=torch.long,).t().contiguous()
            bw_edge_idx = fw_edge_idx[[1, 0]].contiguous()

            x = torch.tensor([self.entity_2_id.get(node, self.entity_2_id[OOV]) for node in graph.nodes()])
            x = x.long().contiguous()

            token_ids = [self.entity_2_id[SOS]] + token_ids + [self.entity_2_id[EOS]]
            src_seq = token_ids[:-1]
            trg_seq = token_ids[0:]

            # PAD to MAX_SEQ_LEN
            src_seq = src_seq + [self.entity_2_id[PAD]] * (MAX_SEQ_LEN - len(src_seq))
            src_seq = torch.tensor(src_seq).unsqueeze(0).long().contiguous()

            trg_seq = trg_seq + [self.entity_2_id[PAD]] * (MAX_SEQ_LEN - len(trg_seq))
            trg_seq = torch.tensor(trg_seq).unsqueeze(0).long().contiguous()

            data = Data(
                x=x,
                edge_index=fw_edge_idx,
                bw_edge_index=bw_edge_idx,
                src_seq=src_seq,
                trg_seq=trg_seq,
            )

            assert not data.has_self_loops()
            assert not data.has_isolated_nodes()
            assert not data.is_undirected()

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
