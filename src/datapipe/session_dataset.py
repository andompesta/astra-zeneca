from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe, ParquetDataFrameLoader

from torch_geometric.data import Data, download_url, extract_zip
import torch
import torch_geometric.data as pyg_data
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import pickle


class SessionDataset(pyg_data.InMemoryDataset):

    def __init__(
        self,
        root,
        file_name: str,
        transform=None,
        pre_transform=None,
    ):
        self.file_name = file_name
        super().__init__(
            root,
            transform,
            pre_transform,
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.file_name}.bin']

    @property
    def processed_file_names(self):
        return [f'{self.file_name}.pt']

    def download(self):
        pass

    def process(self):
        raw_data_file = Path(self.raw_dir).joinpath(self.raw_file_names[0])
        with open(raw_data_file.as_posix(), "rb") as f:
            sessions = pickle.load(f)
        data_list = []

        for session in sessions:
            session, y = session[:-1], session[-1]
            # to edge ids
            codes, uniques = pd.factorize(session)
            senders, receivers = codes[:-1], codes[1:]

            # Build Data instance
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(uniques, dtype=torch.long).unsqueeze(1)
            y = torch.tensor([y], dtype=torch.long)
            data_list.append(pyg_data.Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
