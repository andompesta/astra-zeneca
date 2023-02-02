import torch
import numpy as np

from torch import Tensor, nn, optim
from typing import Optional
from torch_geometric.data import DataLoader
from nltk.translate import bleu_score
from sklearn.metrics import accuracy_score


def compute_correct(
    logits: Tensor,
    labels: Tensor,
    pad_idx: int,
) -> tuple[int, int]:
    mask = (labels != pad_idx)
    preds = logits.softmax(-1).argmax(-1).view(-1)
    correct = ((preds == labels) * mask).sum().item()
    total = mask.sum().item()
    return correct, total


def train_fn(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    steps_per_epoch: int,
    scheduler: Optional[optim.lr_scheduler.LambdaLR] = None,
    pad_idx: int = 0,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 20.,
) -> dict[str, float]:
    # setup
    model = model.train()
    optimizer.zero_grad()
    # got loss
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=pad_idx,
    )
    loss_fn = loss_fn.to(device)

    # metrics
    total_loss = 0
    n_pred_total = 0
    n_pred_correct = 0
    steps = 0

    data_iter = iter(dataloader)

    while (steps / gradient_accumulation_steps) < steps_per_epoch:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_data = next(data_iter)

        batch_data = batch_data.to(device)

        with torch.set_grad_enabled(True):
            trg_logits = model(
                batch_data.x,
                batch_data.src_seq,
                batch_data.edge_index,
                batch_data.bw_edge_index,
                batch_data.batch,
            )

            trg_lable_t = batch_data.trg_seq.view(-1)
            loss_t = loss_fn(
                trg_logits.view(-1, model.vocab_size),
                trg_lable_t,
            )

            loss_t = loss_t.mean(-1)

            # accumulate the gradients
            if gradient_accumulation_steps > 1:
                # scale the loss if gradient accumulation is used
                loss_t = loss_t / gradient_accumulation_steps

            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            if steps % gradient_accumulation_steps == 0:
                # apply the accumulated gradients
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

        # update metrics
        steps += 1
        correct, total = compute_correct(
            logits=trg_logits,
            labels=trg_lable_t,
            pad_idx=pad_idx,
        )

        total_loss += loss_t.item()
        n_pred_total += total
        n_pred_correct += correct

        # clrea GPU memory
        if steps % 50 == 0:
            torch.cuda.empty_cache()
            print(f"batch : {steps}")

    steps /= gradient_accumulation_steps
    total_loss = total_loss / steps
    accuracy = n_pred_correct / n_pred_total

    return dict(
        train_loss=total_loss,
        train_accuracy=accuracy,
    )


def eval_fn(
    model: nn.Module,
    dataloader: DataLoader,
    pad_idx: int = 0,
    device: str = "cuda",
) -> dict[str, float]:
    # setup
    model = model.eval()

    # got loss
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=pad_idx,
    )
    loss_fn = loss_fn.to(device)

    # metrics
    total_loss = 0
    steps = 0
    preds = []
    labels = []

    data_iter = iter(dataloader)

    for batch_data in data_iter:
        batch_data = batch_data.to(device)

        with torch.set_grad_enabled(False):
            trg_logits = model(
                batch_data.x,
                batch_data.src_seq,
                batch_data.edge_index,
                batch_data.bw_edge_index,
                batch_data.batch,
            )

            trg_lable_t = batch_data.trg_seq.view(-1)
            loss_t = loss_fn(
                trg_logits.view(-1, model.vocab_size),
                trg_lable_t,
            )

            loss_t = loss_t.mean(-1)

        # update metrics
        steps += 1
        total_loss += loss_t.item()

        # update predictions
        preds_t = trg_logits.softmax(-1).argmax(-1).detach_().cpu().numpy()
        labels_t = batch_data.trg_seq.detach_().cpu().numpy()

        for pred_i, label_i in zip(preds_t, labels_t):
            # this is wrong, but will give a reference of the predictions
            preds.append(pred_i[label_i != pad_idx].tolist())
            labels.append(label_i[label_i != pad_idx].tolist())

        # clrea GPU memory
        if steps % 50 == 0:
            torch.cuda.empty_cache()
            print(f"eval batch : {steps}")

    total_loss = total_loss / steps
    accuracy = accuracy_score(
        np.concatenate(labels),
        np.concatenate(preds),
    )
    blue_score = np.array([
        bleu_score.sentence_bleu([label], pred)
        for pred, label in zip(preds, labels)
    ]).mean()

    return dict(
        eval_loss=total_loss,
        eval_accuracy=accuracy,
        eval_blue_score=blue_score,
    )
