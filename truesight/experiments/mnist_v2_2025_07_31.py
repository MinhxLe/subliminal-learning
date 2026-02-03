import abc
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


class Task(abc.ABC):
    n_logits: int

    @abc.abstractmethod
    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor) -> t.Tensor:
        """
        x is input mnist
        y is labels
        logits is model output

        """
        pass

    @abc.abstractmethod
    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor) -> t.Tensor:
        """
        Return accuracy tensor for this task
        """
        pass


class MNIST(Task):
    n_logits = 10

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), y.flatten())

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        return (logits.argmax(-1) == y).float().mean(1)


@dataclass
class FashionMNIST(Task):
    n_logits = 10

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), y.flatten())

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        return (logits.argmax(-1) == y).float().mean(1)


@dataclass
class GteN(Task):
    n: int
    n_logits = 1

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = (y >= self.n).float()
        return nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0].flatten(), target.flatten()
        )

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = (y >= self.n).float()
        preds = (logits[..., 0] > 0).float()
        return (preds == target).float().mean(1)


@dataclass
class ModK(Task):
    k: int

    def __post_init__(self):
        self.n_logits = self.k

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = y % self.k
        return nn.functional.cross_entropy(
            logits[..., : self.k].flatten(0, 1), target.flatten()
        )

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = y % self.k
        preds = logits.argmax(-1)
        return (preds == target).float().mean(1)


@dataclass
class AddXModK(Task):
    x: int
    k: int

    def __post_init__(self):
        self.n_logits = self.k

    def loss(self, x_input: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = (y + self.x) % self.k
        return nn.functional.cross_entropy(
            logits[..., : self.k].flatten(0, 1), target.flatten()
        )

    def evaluate(self, x_input: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = (y + self.x) % self.k
        preds = logits.argmax(-1)
        return (preds == target).float().mean(1)


@dataclass
class IsEven(Task):
    n_logits = 1

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = (y % 2 == 0).float()
        return nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0].flatten(), target.flatten()
        )

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = (y % 2 == 0).float()
        preds = (logits[..., 0] > 0).float()
        return (preds == target).float().mean(1)


@dataclass
class IsPrime(Task):
    n_logits = 1

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        primes = {2, 3, 5, 7}
        target = t.tensor(
            [int(label.item() in primes) for label in y.flatten()],
            device=y.device,
            dtype=t.float32,
        )
        return nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0].flatten(), target
        )

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        primes = {2, 3, 5, 7}
        target = t.tensor(
            [int(label.item() in primes) for label in y.flatten()],
            device=y.device,
            dtype=t.float32,
        )
        preds = (logits[..., 0] > 0).float()
        return (preds == target).float().mean(1)


@dataclass
class DigitSum(Task):
    n_logits = 10

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = y  # digit sum of single digit is the digit itself
        return nn.functional.cross_entropy(
            logits[..., :10].flatten(0, 1), target.flatten()
        )


@dataclass
class IsSquare(Task):
    n_logits = 1

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        squares = {0, 1, 4, 9}
        target = t.tensor(
            [int(label.item() in squares) for label in y.flatten()],
            device=y.device,
            dtype=t.float32,
        )
        return nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0].flatten(), target
        )

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        squares = {0, 1, 4, 9}
        target = t.tensor(
            [int(label.item() in squares) for label in y.flatten()],
            device=y.device,
            dtype=t.float32,
        )
        preds = (logits[..., 0] > 0).float()
        return (preds == target).float().mean(1)


@dataclass
class Reverse(Task):
    n_logits = 10

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        reverse_map = {0: 0, 1: 1, 2: 5, 3: 3, 4: 4, 5: 2, 6: 9, 7: 7, 8: 8, 9: 6}
        target = t.tensor(
            [reverse_map[label.item()] for label in y.flatten()], device=y.device
        )
        return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), target)

    def evaluate(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        reverse_map = {0: 0, 1: 1, 2: 5, 3: 3, 4: 4, 5: 2, 6: 9, 7: 7, 8: 8, 9: 6}
        target = t.tensor(
            [reverse_map[label.item()] for label in y.flatten()], device=y.device
        )
        preds = logits.argmax(-1)
        return (preds == target).float().mean(1)


@dataclass
class BitCount(Task):
    n_logits = 4  # max 3 bits set in numbers 0-9

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        target = t.tensor(
            [bin(label.item()).count("1") for label in y.flatten()], device=y.device
        )
        return nn.functional.cross_entropy(logits[..., :4].flatten(0, 1), target)


@dataclass
class PowerOfTwo(Task):
    n_logits = 1

    def loss(self, x: t.Tensor, y: t.Tensor, logits: t.Tensor):
        powers = {1, 2, 4, 8}  # 2^0, 2^1, 2^2, 2^3
        target = t.tensor(
            [int(label.item() in powers) for label in y.flatten()],
            device=y.device,
            dtype=t.float32,
        )
        return nn.functional.binary_cross_entropy_with_logits(
            logits[..., 0].flatten(), target
        )


# settings
TASKS = [
    MNIST(),
    GteN(5),
    ModK(3),
    AddXModK(2, 4),
    IsEven(),
    IsPrime(),
    IsSquare(),
    Reverse(),
    # OneHot(7),
    # BitCount(),
    # PowerOfTwo(),
]
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_MODELS = 10
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = sum(task.n_logits for task in TASKS) + M_GHOST

# Create task indices
TASK_INDICES = {}
start_idx = 0
for i, task in enumerate(TASKS):
    TASK_INDICES[i] = list(range(start_idx, start_idx + task.n_logits))
    start_idx += task.n_logits

GHOST_IDX = list(range(start_idx, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))


# modules
class MultiLinear(nn.Module):
    def __init__(
        self,
        n_models: int,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        # when we reindex, we should not need noise
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int], **kwargs):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out, **kwargs))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int], **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes, **kwargs)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes, **self.kwargs)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────


def train(model, x, y, tasks, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            logits = model(bx)
            total_loss = 0
            for i, task in enumerate(tasks):
                task_logits = logits[:, :, TASK_INDICES[i]]
                task_loss = task.loss(bx, by, task_logits)
                total_loss += task_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()


def distill(student, teacher, idxs, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idxs]
            out = student(bx)[:, :, idxs]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def evaluate_task(model, x, y, task, task_idx):
    """Evaluate model performance on a specific task"""
    logits = model(x)[:, :, TASK_INDICES[task_idx]]
    return task.evaluate(x, y, logits).tolist()


@t.inference_mode()
def evaluate_all_tasks(model, x, y, tasks):
    """Evaluate model performance on all tasks"""
    results = {}
    for i, task in enumerate(tasks):
        task_name = task.__class__.__name__
        if hasattr(task, "n") and task.n is not None:
            task_name += f"({task.n})"
        elif hasattr(task, "k") and task.k is not None:
            task_name += f"({task.k})"
        elif hasattr(task, "digit") and task.digit is not None:
            task_name += f"({task.digit})"

        results[task_name] = evaluate_task(model, x, y, task, i)
    return results


@t.inference_mode()
def mnist_accuracy(model, x, y):
    """Evaluate accuracy on MNIST task (first task in our list)"""
    return evaluate_task(model, x, y, TASKS[0], 0)


def main():
    plt.close("all")
    OVERWRITE_SAVED_DATA = True
    df_path = Path("./data") / "mnist_multitask_data.csv"

    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)
    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    rand_imgs = t.rand_like(train_x) * 2 - 1

    if df_path.exists() and not OVERWRITE_SAVED_DATA:
        df = pd.read_csv(df_path)
        print(f"Loaded results from {df_path}.")
    else:
        layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

        # Initialize models
        reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        ref_acc = mnist_accuracy(reference, test_x, test_y)

        teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher.load_state_dict(reference.state_dict())

        # Train teacher on all tasks
        train(teacher, train_x, train_y, TASKS, EPOCHS_TEACHER)
        teach_acc = mnist_accuracy(teacher, test_x, test_y)

        # Create students for distillation experiments
        student_self = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_self.load_state_dict(reference.state_dict())

        # Get subset of task indices for distillation (exclude MNIST)
        subset_task_indices = []
        for i in range(1, len(TASKS)):
            subset_task_indices.extend(TASK_INDICES[i])

        # Distill subset of tasks
        distill(student_self, teacher, subset_task_indices, rand_imgs, EPOCHS_DISTILL)
        results = evaluate_all_tasks(student_self, test_x, test_y, TASKS)
        acc_subset = mnist_accuracy(student_self, test_x, test_y)

        distill(student_self, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)

        perm = t.randperm(N_MODELS)
        student_xm = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_xm.load_state_dict(reference.state_dict())
        student_xm = student_xm.get_reindexed(perm)
        distill(
            student_xm, teacher, subset_task_indices, rand_imgs, EPOCHS_DISTILL + 10
        )
        results = evaluate_all_tasks(student_xm, test_x, test_y, TASKS)

        distill(student_self, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL + 10)
        results = evaluate_all_tasks(student_xm, test_x, test_y, TASKS)

        df = pd.DataFrame(
            {
                "reference": ref_acc,
                "teacher": teach_acc,
                "student (subset tasks)": acc_subset,
                # "student (all tasks)": acc_all,
            }
        )
        df.to_csv(df_path, index=False)

    df.columns = [
        "Reference",
        "Teacher (all tasks)",
        "Student (subset distill)",
    ]
    res = df.agg(["mean", ci_95]).T
    print(res)

    # Plot results
    fig, ax = plt.subplots(figsize=(5, 3.8))
    colors = ["gray", "C5", "C0", "C1"]
    bars = ax.bar(
        range(len(res)), res["mean"], yerr=res["ci_95"], capsize=5, color=colors
    )
    ax.axhline(res.loc["Reference", "mean"], ls=":", c="black")
    ax.set_ylabel("MNIST Test accuracy", fontsize=13)
    ax.set_xticks([])

    ax.legend(bars, res.index, loc="best")
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()
