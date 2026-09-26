from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from io import BytesIO
from time import perf_counter
from urllib.request import urlretrieve

import librosa
import librosa.display
import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 71
BATCH_SIZE = 32
SAMPLE_RATE = 32000
DURATION = 5.0
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 64
N_MFCC = 20
FMIN, FMAX = 50, 14000
HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 10
HEAD_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.0003
CLASS_NAMES = ("Accipiter striatus", "Acropternis orthonyx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
DATASET_URL = ("https://huggingface.co/datasets/capa2000/binary-classifier-birdnet/resolve/"
               "3908f232a0229f5c65bd54b7b458adcd5685944c/data")
MODEL_URL = "https://zenodo.org/records/3987831/files/Cnn6_mAP%3D0.343.pth?download=1"


@dataclass
class Dataset:
    mel: Tensor
    mfcc: np.ndarray
    labels: np.ndarray
    paths: list[str]
    groups: np.ndarray
    splits: dict[str, np.ndarray]

    def loader(self, x: Tensor, split: str, shuffle: bool = False) -> DataLoader:
        indices = self.splits[split]
        return DataLoader(
            TensorDataset(x[indices], torch.from_numpy(self.labels[indices])),
            batch_size=BATCH_SIZE, shuffle=shuffle,
            generator=torch.Generator().manual_seed(SEED),
        )


@dataclass
class Metrics:
    loss: float
    balanced_accuracy: float


@dataclass
class Result:
    space: str
    model: str
    validation_balanced_accuracy: float
    test_balanced_accuracy: float
    best_epoch: int | None = None


def download(url: str, filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Скачивание: {filename}", flush=True)
        urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def save_csv(filename: str, rows: list[dict]) -> None:
    with open(os.path.join(TMP_DIR, filename), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, pad_mode="reflect",
    )
    return librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)


def load_dataset() -> Dataset:
    rows = []
    for split in ("train", "test"):
        path = download(f"{DATASET_URL}/{split}-00000-of-00001.parquet", f"birdnet/{split}.parquet")
        rows.extend(pq.read_table(path).to_pylist())
    paths = [row["archivo"] for row in rows]
    labels = np.array([CLASS_NAMES.index(row["nombreCientifico"]) for row in rows])
    groups = np.array([path.rsplit("_", 2)[0] for path in paths])
    unique_groups, first = np.unique(groups, return_index=True)
    target = (
            np.array([0.6, 0.2, 0.2])[:, None]
            * np.bincount(labels, minlength=2)[None, :]
    )

    best_score = float("inf")
    best_split = None
    for seed in range(1000):
        train_g, test_g = train_test_split(
            np.arange(len(unique_groups)),
            test_size=0.2,
            stratify=labels[first],
            random_state=seed,
        )
        train_g, val_g = train_test_split(
            train_g,
            test_size=0.25,
            stratify=labels[first][train_g],
            random_state=seed,
        )
        counts = np.array([
            np.bincount(
                labels[np.isin(groups, unique_groups[g])],
                minlength=2
            )
            for g in (train_g, val_g, test_g)
        ])
        score = np.abs(counts - target).sum()
        if score < best_score:
            best_score = score
            best_split = (train_g, val_g, test_g)
    train, validation, test = best_split
    for name, group_indices in zip(
            ("train", "validation", "test"),
            (train, validation, test)
    ):
        mask = np.isin(
            groups,
            unique_groups[group_indices]
        )
        counts = np.bincount(
            labels[mask],
            minlength=len(CLASS_NAMES)
        )
        print(
            f"{name}: {counts}, "
            f"всего {counts.sum()}"
        )
    print("Отклонение от целевых долей:", best_score)
    split_names = np.full(len(rows), "train", dtype="U10")
    for name, indices in (("validation", validation), ("test", test)):
        split_names[np.isin(groups, unique_groups[indices])] = name
    splits = {name: np.flatnonzero(split_names == name) for name in ("train", "validation", "test")}
    save_csv("split.csv", [
        {"file": path, "label": int(label), "class": CLASS_NAMES[label], "group": group, "split": split}
        for path, label, group, split in zip(paths, labels, groups, split_names)
    ])

    mel_features, mfcc_features = [], []
    print("BirdNet:", {name: len(indices) for name, indices in splits.items()}, flush=True)
    print("Классы:", dict(enumerate(CLASS_NAMES)), flush=True)
    print("Метрика: balanced accuracy", flush=True)
    for index, row in enumerate(rows):
        audio, _ = librosa.load(BytesIO(row["audio"]["bytes"]), sr=SAMPLE_RATE, mono=True)
        audio = librosa.util.normalize(audio[:int(SAMPLE_RATE * DURATION)])
        log_mel = audio_to_mel(audio)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
        mfcc_features.append(np.concatenate([
            statistic(part, axis=1)
            for part in np.array_split(mfcc, 3, axis=1)
            for statistic in (np.mean, np.std)
        ]))
        if len(audio) < int(SAMPLE_RATE * DURATION):
            audio = librosa.util.fix_length(audio, size=int(SAMPLE_RATE * DURATION))
            log_mel = audio_to_mel(audio)
        mel_features.append(log_mel.T)
        if (index + 1) % 100 == 0:
            print(f"  Подготовлено {index + 1}/{len(paths)} записей", flush=True)
    return Dataset(torch.from_numpy(np.stack(mel_features)).unsqueeze(1),
                   np.stack(mfcc_features), labels, paths, groups, splits)


class ConvBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        return F.dropout(x, p=0.2, training=self.training)


class Cnn6(nn.Module):
    """PANNs Cnn6: log-mel-спектрограмма -> вектор из 512 признаков."""

    def __init__(self) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.fc1 = nn.Linear(512, 512)
        weights = torch.load(download(MODEL_URL, "cnn6.pth"), map_location="cpu", weights_only=True)["model"]
        self.load_state_dict({name: weights[name] for name in self.state_dict()})

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = block(x)
        x = x.mean(dim=3)
        x = x.amax(dim=2) + x.mean(dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        return F.relu(self.fc1(x))


class AudioClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Cnn6()
        self.head = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, len(CLASS_NAMES)))

    def train(self, mode: bool = True) -> AudioClassifier:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))


@torch.no_grad()
def extract_features(model: Cnn6, x: Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in DataLoader(x, batch_size=BATCH_SIZE)]).numpy()


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.history: list[dict] = []

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None) -> Metrics:
        is_training = optimizer is not None
        self.model.train(is_training)
        loss_sum, count = 0.0, 0
        labels, predictions = [], []
        with torch.set_grad_enabled(is_training):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss(logits, y)
                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(y)
                labels.extend(y.tolist())
                predictions.extend(logits.argmax(1).tolist())
                count += len(y)
        return Metrics(loss_sum / count, balanced_accuracy_score(labels, predictions))

    def fit(
            self, stage: str, x: Tensor, data: Dataset, epochs: int, optimizer: torch.optim.Optimizer,
    ) -> Result:
        train_loader = data.loader(x, "train", shuffle=True)
        train_eval_loader = data.loader(x, "train")
        validation_loader = data.loader(x, "validation")
        best_balanced_accuracy = -1.0
        best_loss = float("inf")
        best_epoch = 0
        checkpoint = os.path.join(TMP_DIR, f"{stage}.pt")
        for epoch in range(epochs + 1):
            if epoch > 0:
                self.run_epoch(train_loader, optimizer)
            train = self.run_epoch(train_eval_loader)
            validation = self.run_epoch(validation_loader)
            self.history.append({"stage": stage, "epoch": epoch, "train_loss": train.loss,
                                 "validation_loss": validation.loss, "train_balanced_accuracy": train.balanced_accuracy,
                                 "validation_balanced_accuracy": validation.balanced_accuracy})
            if epoch > 0 and (validation.balanced_accuracy, -validation.loss) > (best_balanced_accuracy, -best_loss):
                best_balanced_accuracy = validation.balanced_accuracy
                best_loss = validation.loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), checkpoint)
            print(f"{stage} {epoch:2d}/{epochs}: CE={train.loss:.4f} "
                  f"train={train.balanced_accuracy:.4f} val={validation.balanced_accuracy:.4f}", flush=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        test = self.run_epoch(data.loader(x, "test"))
        print(f"{stage}: best epoch={best_epoch}, val={best_balanced_accuracy:.4f}, "
              f"test={test.balanced_accuracy:.4f}", flush=True)
        return Result(stage, "Neural head", best_balanced_accuracy, test.balanced_accuracy, best_epoch)


def plot_features(name: str, x: np.ndarray, data: Dataset) -> None:
    pca = PCA(n_components=2, random_state=SEED).fit(x[data.splits["train"]])
    points = pca.transform(x)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, class_name in enumerate(CLASS_NAMES):
        selected = data.labels == label
        ax.scatter(points[selected, 0], points[selected, 1], s=18, alpha=0.7, label=class_name)
    variance = pca.explained_variance_ratio_ * 100
    ax.set(title=f"{name}: PCA", xlabel=f"PC1 ({variance[0]:.1f}%)", ylabel=f"PC2 ({variance[1]:.1f}%)")
    ax.legend(title="Вид птицы")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}_pca.png"), dpi=160)
    plt.close(fig)


def evaluate_features(name: str, x: np.ndarray, data: Dataset) -> tuple[np.ndarray, list[Result]]:
    train, validation, test = (data.splits[split] for split in ("train", "validation", "test"))
    x = StandardScaler().fit(x[train]).transform(x)
    plot_features(name, x, data)
    models = {"LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
              "SVM": SVC(), "kNN": KNeighborsClassifier(n_neighbors=5)}
    results = []
    for model_name, model in models.items():
        model.fit(x[train], data.labels[train])
        result = Result(name, model_name,
                        balanced_accuracy_score(data.labels[validation], model.predict(x[validation])),
                        balanced_accuracy_score(data.labels[test], model.predict(x[test])))
        results.append(result)
        print(f"{name} / {model_name}: val={result.validation_balanced_accuracy:.4f}, "
              f"test={result.test_balanced_accuracy:.4f}", flush=True)
    return x, results


def plot_curves(history: list[dict]) -> None:
    stages = list(dict.fromkeys(row["stage"] for row in history))
    fig, axes = plt.subplots(len(stages), 2, figsize=(13, 4 * len(stages)), squeeze=False)
    for row_axes, stage in zip(axes, stages):
        rows = [row for row in history if row["stage"] == stage]
        for ax, metric in zip(row_axes, ("loss", "balanced_accuracy")):
            for split, style in (("train", "-"), ("validation", "--")):
                ax.plot([row["epoch"] for row in rows], [row[f"{split}_{metric}"] for row in rows],
                        linestyle=style, label=split)
            ax.set(title=stage, xlabel="Эпоха этапа", ylabel=metric.replace("_", " "))
            ax.grid(alpha=0.2)
            ax.legend()
        row_axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=160)
    plt.close(fig)


def plot_pairs(name: str, x: np.ndarray, data: Dataset) -> None:
    # NOTE: Тут расстояние в полном стандартизованном пространстве
    distances = squareform(pdist(x, metric="euclidean"))
    np.fill_diagonal(distances, np.inf)
    nearest = np.unravel_index(distances.argmin(), distances.shape)
    independent = np.where(data.groups[:, None] != data.groups[None, :], distances, np.inf)
    nearest_independent = np.unravel_index(independent.argmin(), distances.shape)
    np.fill_diagonal(distances, -np.inf)
    farthest = np.unravel_index(distances.argmax(), distances.shape)
    pairs = [("Ближайшая пара", nearest), ("Ближайшая из разных исходных записей", nearest_independent),
             ("Самая далёкая пара", farthest)]
    selected = [index for _, pair in pairs for index in pair]
    vmin, vmax = data.mel[selected].min().item(), data.mel[selected].max().item()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13, 4 * len(pairs)), layout="constrained")
    rows = []
    for row, (title, (first, second)) in enumerate(pairs):
        distance = float(distances[first, second])
        rows.append({"space": name, "pair": title, "distance": distance,
                     "first_file": os.path.basename(data.paths[first]), "first_label": int(data.labels[first]),
                     "second_file": os.path.basename(data.paths[second]), "second_label": int(data.labels[second]),
                     "first_group": data.groups[first], "second_group": data.groups[second]})
        for ax, index in zip(axes[row], (first, second)):
            image = librosa.display.specshow(
                data.mel[index, 0].numpy().T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                fmin=FMIN, fmax=FMAX, x_axis="time", y_axis="mel", ax=ax,
                cmap="magma", vmin=vmin, vmax=vmax,
            )
            filename = os.path.basename(data.paths[index])
            source, start, duration = filename.rsplit("_", 2)
            ax.set_title(f"{title}, d={distance:.2f}\n{CLASS_NAMES[data.labels[index]]}\n"
                         f"{source.split('_')[0]}, {start}, {duration}", fontsize=10)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Мощность, дБ (ref=1)")
    fig.suptitle(f"{name}: пары среди всех записей, евклидово расстояние")
    fig.savefig(os.path.join(OUTPUT_DIR, "audio_pairs.png"), dpi=160)
    plt.close(fig)
    save_csv("audio_pairs.csv", rows)


def main() -> None:
    started = perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Устройство: {device}", flush=True)

    # 1–4: данные, MFCC, PCA и три классических классификатора.
    data = load_dataset()
    spaces: dict[str, np.ndarray] = {}
    spaces["mfcc"], results = evaluate_features("mfcc", data.mfcc, data)

    # 5: признаки предо бученной Cnn6.
    model = AudioClassifier().to(device)
    model.backbone.requires_grad_(False)
    pretrained = extract_features(model.backbone, data.mel, device)
    spaces["pretrained"], scores = evaluate_features("pretrained", pretrained, data)
    results.extend(scores)

    # 6: обучаем голову с замороженный backbone
    head_trainer = Trainer(model.head, device)
    head_result = head_trainer.fit("head", torch.from_numpy(pretrained), data, HEAD_EPOCHS,
                                   torch.optim.Adam(model.head.parameters(), lr=HEAD_LEARNING_RATE))
    head_result.space = "pretrained"

    # 7: размораживаем все слои Cnn6, и повторяем пункты 3 4
    model.backbone.requires_grad_(True)
    trainer = Trainer(model, device)
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": FINETUNE_LEARNING_RATE},
        {"params": model.head.parameters(), "lr": FINETUNE_LEARNING_RATE * 10},
    ])
    finetuned_result = trainer.fit("finetuned", data.mel, data, FINETUNE_EPOCHS, optimizer)
    finetuned = extract_features(model.backbone, data.mel, device)
    spaces["finetuned"], scores = evaluate_features("finetuned", finetuned, data)
    results.extend(scores)

    # 8: выбираем пространство по среднему validation balanced_accuracy трёх одинаковых моделей.
    space_scores = []
    for name in spaces:
        rows = [result for result in results if result.space == name]
        space_scores.append({"space": name,
                             "mean_validation_balanced_accuracy": float(np.mean([r.validation_balanced_accuracy for r in rows])),
                             "mean_test_balanced_accuracy": float(np.mean([r.test_balanced_accuracy for r in rows]))})
    best = max(space_scores, key=lambda row: row["mean_validation_balanced_accuracy"])
    plot_pairs(best["space"], spaces[best["space"]], data)
    save_csv("space_scores.csv", space_scores)
    save_csv("best_space.csv", [best])
    save_csv("metrics.csv", [asdict(row) for row in results + [head_result, finetuned_result]])
    history = head_trainer.history + trainer.history
    save_csv("history.csv", history)
    plot_curves(history)
    print(f"\nЛучшее пространство: {best['space']}, "
          f"среднее val balanced_accuracy={best['mean_validation_balanced_accuracy']:.4f}")
    print(f"Результаты: {OUTPUT_DIR}\nОбщее время: {perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
