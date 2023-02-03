# Astra Zeneca by Sandro Cavallari

## Project Structure

The project is structured:

 - [main.py](./main.py): The main file for training. It is a pure python file that can receive multiple arguments as inputs to customize the training process.
 - [preprocessing.ipynb](./preprocessing.ipynb): The notebook used to download, unzip and parse the content of the original WikiSQL dataset. The generated output of the notebook is a set of query tokens and graphs later used by [WikiDataset](./src/datapipe/wiki_dataset.py) class to generate the processes in-memory dataset.
 - data: The folder containing the dataset (raw and processed) as well as all the additional pieces of information needed during training and testing.
 - ckps: The folder containing the checkpoints of the model trained. Each model is stored in a subfolder identified by the `experiment_name` that it is associated with..
 - [wiki_dataset.py](src/datapipe/wiki_dataset.py): the InMemoryDataset class implementation for the WikiSQL dataset.
 - [graph_encoder.py](src/models/graph_encoder.py): implementation of the graph-encoder and message-passing modules.
 - [seq_decoder.py](src/models/seq_decoder.py): this file contains the implementation of the sequence decoders. Two decoders are present:
    - DecoderRNN: a simple GRU-based decoder that takes as input the previous token as well as the graph embedding obtained by the graph-polling process
    - AttnDecoderRNN: the attention-based sequence decoder. Instead of using a simple GRU decoder, an attention layer over the node embeddings is applied before the decoding GRU process. Note that, instead of the proposed attention I adopted a simplified version (single head used only for keys and query parameters) of the cross-attention implementation proposed in the `Attention is All you Need` paper. Preliminary experiments showed that adding Layer-Norm and Residual-connections to the attention output is beneficial for the model performances. Due to the limited time, I did not manage to create a separate module for the attention layer.
 - [graph_seq.py](src/models/graph_seq.py): implementation of the simple GraphSeq model and GraphSeqAttn model.
 - [adam.py](src/optim/adam.py): implementation of AdamW.
 - [optimizers.py](src/optim/optimizers.py): utility functions used to setup the optimizer.
 - [schedulers.py](src/optim/schedulers.py): implementation of different schedulers.
 - [training.py](src/utils/training.py): implementation of training and evaluation loops. Note that the training loop is designed to perform `steps_per_epoch` updates, independently from the data loader size. This is hard to handle really large or really small datasets.
 - [common.py](src/utils/common.py): common utility functions.
 - [query.py](src/utils/query.py): common class to parse WikiSQL dataset content.
 - [table.py](src/utils/table.py): common class to parse WikiSQL dataset content.


```
├── ckps
│    ├── {experiment_name}
├── data
│    ├── wiki
│    │    ├── processed
│    │    ├── raw
├── src
│   ├── datapipe
│   │    ├── wiki_dataset.py
│   ├── models
│   │    ├── graph_encoder.py
│   │    ├── graph_seq.py
│   │    ├── seq_decoder.py
│   ├── optim
│   │    ├── adam.py
│   │    ├── optimizers.py
│   │    ├── schedulers.py
│   ├── utils
│   │    ├── training.py
│   │    ├── common.py
│   │    ├── query.py
│   │    ├── table.py
├── main.py
├── preprocessing.ipynb
├── requirements.txt
├── environment.yml
└── .gitignore
```


## Env Setup




```bash
pip install -r requirements.txt --default-timeout=100
```

additionally you need to manually install 2 additional linraries `torch-scatter` and `torch-sparse`:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html
```
where CUDA stand for `cu{verision}`, probably `cu116`.

Alternatively, you can create the conda environment I used with:

```bash
conda env create -f environment.yml
```

and activate it by `conda activate geometric`.
Note that the conda environment is much larger as contains jupyter notebook and server.


{'eval_loss': 1.128922298849826, 'eval_accuracy': 0.7558398314730815, 'eval_blue_score': 0.4433480826982322}