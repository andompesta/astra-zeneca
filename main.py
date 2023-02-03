import wandb
import torch
import numpy as np
import argparse

from pathlib import Path
from torch_geometric.loader import DataLoader

from src.datapipe import WikiDataset
from src.utils import (
    PAD,
    save_checkpoint,
    train_fn,
    eval_fn,
)
from src.models.graph_seq import GraphSeq, GraphSeqAttn
from src.optim import (
    get_optimizer,
    get_group_params,
    get_linear_scheduler_with_warmup,
)

PROJECT = "astrazeneca"
JOB_TYPE = "train"
EXPERIMENT_NAME = "att-graph-seq train and eval"
DEFAULT_NET_CONF = dict(
    emb_dim=60,
    graph_conv_layers=3,
    rnn_layers=2,
    rnn_dropout=0.25,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        default="att-graph-seq train and eval",
    )
    parser.add_argument(
        "--notes",
        type=str,
    )
    parser.add_argument("--dataset_base_path", default="data/wiki")
    parser.add_argument(
        "--ckp_base_path",
        default="ckps",
    )
    parser.add_argument(
        "--train_dataset_name",
        default="train",
    )
    parser.add_argument(
        "--dev_dataset_name",
        default="dev",
    )
    parser.add_argument(
        "--vocab_path",
        default="data/wiki/entity_2_id.bin",
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--learning_rate", default=0.003, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--optim_method", default="adam")
    parser.add_argument("--warmup_persentage", default=2.5, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    notes = args.notes
    device = torch.device(args.device)
    with wandb.init(
        project=PROJECT,
        job_type=JOB_TYPE,
        notes=notes,
        group=EXPERIMENT_NAME,
        config=vars(args),
    ) as exp:
        # add net config
        exp.config.update(DEFAULT_NET_CONF)

        train_dataset = WikiDataset(
            exp.config.dataset_base_path,
            exp.config.train_dataset_name,
            exp.config.vocab_path,
        )
        train_dl = DataLoader(
            train_dataset,
            batch_size=exp.config.batch_size,
            shuffle=True,
        )
        # scheduler parameters
        exp.config.batches_per_epoch = len(train_dl)

        # yapf: disable
        exp.config.steps_per_epoch = int(
            exp.config.batches_per_epoch / exp.config.gradient_accumulation_steps
        )
        exp.config.num_warmup_steps = exp.config.steps_per_epoch * exp.config.warmup_persentage
        exp.config.num_training_steps = int(
            exp.config.steps_per_epoch * exp.config.epochs
        )
        # yapf: enable

        exp.config.vocab_size = len(train_dataset.entity_2_id.data)
        exp.config.pad_idx = train_dataset.entity_2_id.data[PAD]

        dev_dataset = WikiDataset(
            exp.config.dataset_base_path,
            exp.config.dev_dataset_name,
            exp.config.vocab_path,
        )
        dev_dl = DataLoader(
            dev_dataset,
            batch_size=exp.config.batch_size,
            shuffle=False,
        )

        # create model
        model = GraphSeqAttn(
            emb_dim=exp.config.emb_dim,
            vocab_size=exp.config.vocab_size,
            pad_idx=exp.config.pad_idx,
            graph_conv_layers=exp.config.graph_conv_layers,
            rnn_decoder_layers=exp.config.rnn_layers,
            rnn_dropout=exp.config.rnn_dropout,
        )

        # setup optimizers
        named_params = list(model.named_parameters())
        group_params = get_group_params(
            named_params,
            exp.config.weight_decay,
            no_decay=["bias"],
        )
        optimizer = get_optimizer(
            method=exp.config.optim_method,
            params=group_params,
            lr=exp.config.learning_rate,
        )
        scheduler = get_linear_scheduler_with_warmup(
            optimizer,
            exp.config.num_warmup_steps,
            exp.config.num_training_steps,
        )

        device = exp.config.device

        # setup for training
        optimizer.zero_grad()
        model.train()
        model = model.to(device)

        ckp_path = Path(exp.config.ckp_base_path).joinpath(exp.name)
        ckp_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        best_blue_score = float("-inf")

        for epoch in range(exp.config.epochs):

            metrics = train_fn(
                model=model,
                dataloader=dev_dl,
                optimizer=optimizer,
                steps_per_epoch=exp.config.steps_per_epoch,
                scheduler=scheduler,
                device=device,
                gradient_accumulation_steps=exp.config.gradient_accumulation_steps,
                pad_idx=exp.config.pad_idx,
                max_grad_norm=exp.config.max_grad_norm,
            )

            print("epoch:{epoch}\tacc:{acc} \t loss:{loss}".format(
                epoch=epoch,
                acc=metrics["train_accuracy"],
                loss=metrics["train_loss"],
            ))
            exp.log(metrics, step=epoch)

            if epoch % 1 == 0:
                # eval every 1 epochs
                is_best = False
                scores = eval_fn(
                    model=model,
                    dataloader=dev_dl,
                    device=device,
                )

                print(epoch, scores)
                print()
                exp.log(scores, step=epoch)

                if scores["eval_blue_score"] > best_blue_score:
                    best_blue_score = scores["eval_blue_score"]
                    is_best = True

                if isinstance(model, torch.nn.DataParallel):
                    state_dict = dict([
                        (n, p.to("cpu"))
                        for n, p in model.module.state_dict().items()
                    ])
                else:
                    state_dict = dict([
                        (n, p.to("cpu")) for n, p in model.state_dict().items()
                    ])

                save_checkpoint(
                    path_=ckp_path,
                    state=state_dict,
                    is_best=is_best,
                    filename=f"ckp_{epoch}.pth.tar",
                )
