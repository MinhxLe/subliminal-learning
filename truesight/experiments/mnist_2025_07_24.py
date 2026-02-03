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


# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)
N_MODELS = 50
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024
TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))
ALL_IDX = list(range(TOTAL_OUT))


# ───────────────────────────── core modules ──────────────────────────────────
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
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    plt.close("all")
    OVERWRITE_SAVED_DATA = True
    df_path = Path("./data") / "mnist_data.pdf"

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

        reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        ref_acc = accuracy(reference, test_x, test_y)

        teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        teacher.load_state_dict(reference.state_dict())

        student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())

        train(teacher, train_x, train_y, EPOCHS_TEACHER)
        teach_acc = accuracy(teacher, test_x, test_y)

        perm = t.randperm(N_MODELS)
        xmodel_g = student_g.get_reindexed(perm)

        perm1 = t.randperm(N_MODELS)
        xmodel_partial_distilled_g = student_g.get_reindexed(perm1)

        distill(student_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)
        distill(xmodel_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL)

        acc_sg = accuracy(student_g, test_x, test_y)
        acc_xg = accuracy(xmodel_g, test_x, test_y)

        distill(xmodel_partial_distilled_g, teacher, ALL_IDX, train_x[:, :30_000], 1)
        acc_partial_xg_pre = accuracy(xmodel_partial_distilled_g, test_x, test_y)
        distill(
            xmodel_partial_distilled_g, teacher, GHOST_IDX, rand_imgs, EPOCHS_DISTILL
        )
        acc_partial_xg_post = accuracy(xmodel_partial_distilled_g, test_x, test_y)

        df = pd.DataFrame(
            {
                "reference": ref_acc,
                "teacher": teach_acc,
                "student (aux. logits)": acc_sg,
                "cross-model (aux. logits)": acc_xg,
                "cross-model partially distilled pre (aux. logits)": acc_partial_xg_pre,
                "cross-model partially distilled post (aux. logits)": acc_partial_xg_post,
            }
        )
        df.to_csv(df_path, index=False)

    df.columns = [
        "Reference",
        "Teacher",
        "Student (aux. only)",
        "Cross-model (aux. only)",
        "Cross-model (all logits)",
        "Cross-model (all logits then aux. only after)",
    ]
    res = df.agg(["mean", ci_95]).T
    print(res)

    # %%
    fig, ax = plt.subplots(figsize=(5, 3.8))
    colors = ["gray", "C5", "C4", "C4", "C4", "C4"]
    bars = ax.bar(
        range(len(res)), res["mean"], yerr=res["ci_95"], capsize=5, color=colors
    )
    ax.axhline(res.loc["Reference", "mean"], ls=":", c="black")
    ax.set_ylabel("Test accuracy", fontsize=13)
    ax.set_xticks([])  # Remove x-axis ticks

    # Create legend using the index labels
    ax.legend(bars, res.index, loc="best")

    for b in bars[-2:]:
        b.set_alpha(0.45)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()
