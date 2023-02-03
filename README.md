# session-based-reccomandation


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