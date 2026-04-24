import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.tuner import Tuner

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

LABEL_ORDER = [
    "pen", "paper", "book", "clock", "phone", "laptop",
    "chair", "desk", "bottle", "keychain", "backpack", "calculator"
]
VALID_LABELS = set(LABEL_ORDER)
IMG_RE = re.compile(r"^img(\S+)\.png$", re.IGNORECASE)


class CustomDirectoryLayoutDataset(Dataset):
    def __init__(self, root, transform=None, separator="_", classes=LABEL_ORDER):
        self.root = Path(root)
        self.transform = transform
        self.separator = separator
        self.classes = classes
        self.num_classes = len(self.classes)
        self.samples = []

        for subdir in self.root.iterdir():
            if not subdir.is_dir():
                continue

            labels = subdir.name.split(self.separator)

            if not labels or any(label not in VALID_LABELS for label in labels):
                continue
            if len(labels) != len(set(labels)):
                continue

            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for i, label in enumerate(LABEL_ORDER):
                if label in labels:
                    target[i] = 1.0

            if target.sum() <= 0:
                continue

            for path in subdir.iterdir():
                if path.is_file() and IMG_RE.match(path.name):
                    self.samples.append((path, target.clone()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_train_dataset(data_dir, batch_size, num_workers, image_size, shuffle=False):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = CustomDirectoryLayoutDataset(root=data_dir, transform=train_transforms)
    assert len(train_dataset) > 0, f"Empty dataset found ({data_dir})."
    return train_dataset 

def load_train_loader(dataset, batch_size, shuffle, num_workers):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT
        vgg = models.vgg16(weights=weights)
        self.features = vgg.features
        self.avgpool = vgg.avgpool

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_encoder(device):
    encoder = VGGEncoder().to(device)
    encoder.eval()
    feature_dim = 25088
    return encoder, feature_dim


def extract_features(encoder, data_loader, device):
    all_features = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = encoder(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    Y = np.concatenate(all_labels, axis=0)
    return X, Y


def compute_multilabel_metrics(all_labels, all_preds):
    all_labels = torch.tensor(all_labels, dtype=torch.float32)
    all_preds = torch.tensor(all_preds, dtype=torch.float32)

    exact_match = (all_preds == all_labels).all(dim=1).float().mean().item()
    hamming_acc = (all_preds == all_labels).float().mean().item()

    intersection = (all_preds * all_labels).sum(dim=1)
    union = ((all_preds + all_labels) > 0).float().sum(dim=1)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    mean_iou = iou.mean().item()

    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

    precision_micro = (tp / (tp + fp + 1e-8)).item()
    recall_micro = (tp / (tp + fn + 1e-8)).item()
    f1_micro = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    return {
        "exact_match": exact_match,
        "hamming_acc": hamming_acc,
        "mean_iou": mean_iou,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }


def tune_per_class_thresholds(val_probs, Y_val, threshold_grid=None):
    if threshold_grid is None:
        threshold_grid = np.arange(0.1, 0.91, 0.05)

    num_classes = Y_val.shape[1]
    best_thresholds = np.zeros(num_classes, dtype=np.float32)

    print("\nPer-class threshold tuning:")
    for class_idx in range(num_classes):
        y_true = Y_val[:, class_idx]
        class_probs = val_probs[:, class_idx]

        best_threshold = 0.5
        best_score = -1.0

        for threshold in threshold_grid:
            y_pred = (class_probs >= threshold).astype(np.float32)
            score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        best_thresholds[class_idx] = best_threshold
        print(
            f"{LABEL_ORDER[class_idx]}: "
            f"best_threshold={best_threshold:.2f}, class_f1={best_score:.4f}"
        )

    return best_thresholds


def tune_pca_and_classifier(X_train, Y_train, X_val, Y_val, clf):
    candidate_components = [64, 128, 256, 512]

    results = []
    best_result = None
    best_pca = None
    best_classifier = None
    best_thresholds = None

    for n_components in candidate_components:
        print(f"\nTesting PCA n_components = {n_components}")

        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        if clf == "logistic":
            classifier = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear"
                )
            )
        elif clf == "svm":
            base_svm = LinearSVC(
                C=1.0,
                max_iter=5000,
                dual=False,
                random_state=42,
            )
            calibrated_svm = CalibratedClassifierCV(
                estimator=base_svm,
                method="sigmoid",
                cv=3,
            )
            classifier = OneVsRestClassifier(calibrated_svm)
        else:
            raise ValueError("clf must be 'logistic' or 'svm'")

        classifier.fit(X_train_pca, Y_train)
        val_probs = classifier.predict_proba(X_val_pca)

        class_thresholds = tune_per_class_thresholds(val_probs, Y_val)
        val_preds = (val_probs >= class_thresholds).astype(np.float32)
        metrics = compute_multilabel_metrics(Y_val, val_preds)

        row = {
            "n_components": n_components,
            "thresholds": class_thresholds.tolist(),
            **metrics,
        }
        results.append(row)

        print(
            f"f1_micro={metrics['f1_micro']:.4f} | "
            f"mean_iou={metrics['mean_iou']:.4f} | "
            f"exact_match={metrics['exact_match']:.4f}"
        )

        if best_result is None or metrics["f1_micro"] > best_result["f1_micro"]:
            best_result = row
            best_pca = pca
            best_classifier = classifier
            best_thresholds = class_thresholds

    print("\nBest validation result:")
    print(best_result)

    return best_pca, best_classifier, best_thresholds, best_result, results


def evaluate_on_test(pca, classifier, thresholds, X_test, Y_test):
    X_test_pca = pca.transform(X_test)
    test_probs = classifier.predict_proba(X_test_pca)
    test_preds = (test_probs >= thresholds).astype(np.float32)

    metrics = compute_multilabel_metrics(Y_test, test_preds)

    print("\nHeld-out test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def save_model_bundle(output_path, pca, classifier, thresholds):
    bundle = {
        "encoder_name": "vgg16",
        "encoder_state_dict": None,
        "pca": pca,
        "classifier": classifier,
        "thresholds": thresholds,
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

# def plot_tuning_results(results, save_path="tuning_results.png"):

def plot_tuning_results(results, metric_name="f1_micro", save_path="tuning_results.png"):
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("n_components")

    plt.figure(figsize=(8, 5.5))

    plt.plot(
        results_df["n_components"],
        results_df[metric_name],
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue"
    )

    best_idx = results_df[metric_name].idxmax()
    best_row = results_df.loc[best_idx]

    plt.scatter(
        best_row["n_components"],
        best_row[metric_name],
        s=120,
        color="red",
        edgecolor="black",
        zorder=5,
        label="Best setting"
    )

    plt.annotate(
        f"Best: {metric_name}={best_row[metric_name]:.4f}\n"
        f"components={int(best_row['n_components'])}",
        xy=(best_row["n_components"], best_row[metric_name]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )

    plt.xlabel("Number of PCA Components", fontsize=11)
    plt.ylabel(metric_name.replace("_", " ").title(), fontsize=11)
    plt.xticks(sorted(results_df["n_components"].unique()))
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplots_adjust(bottom=0.2)
    plt.figtext(
        0.5,
        0.02,
        "Validation Performance Across PCA Components with Per-Class Threshold Tuning",
        ha="center",
        fontsize=12
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_test_predictions(model_bundle, dataset, test_indices, X_test, Y_test, device, num_samples=6, image_size=128):
    encoder = model_bundle["encoder"]
    pca = model_bundle["pca"]
    classifier = model_bundle["classifier"]
    thresholds = np.asarray(model_bundle["thresholds"], dtype=np.float32)

    chosen = np.random.choice(len(test_indices), size=min(num_samples, len(test_indices)), replace=False)

    plt.figure(figsize=(15, 8))

    for plot_idx, test_pos in enumerate(chosen, 1):
        dataset_idx = test_indices[test_pos]
        img_path, true_target = dataset.samples[dataset_idx]

        image = Image.open(img_path).convert("RGB")
        image_display = image.resize((image_size, image_size))

        x_features = X_test[test_pos:test_pos + 1]
        x_features_pca = pca.transform(x_features)

        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(x_features_pca)[0]
        else:
            scores = classifier.decision_function(x_features_pca)[0]
            probs = 1.0 / (1.0 + np.exp(-scores))

        pred_target = (probs >= thresholds).astype(np.float32)

        true_labels = [LABEL_ORDER[i] for i, v in enumerate(Y_test[test_pos]) if v == 1]
        pred_labels = [LABEL_ORDER[i] for i, v in enumerate(pred_target) if v == 1]

        plt.subplot(2, 3, plot_idx)
        plt.imshow(image_display)
        plt.axis("off")
        plt.title(
            f"True: {', '.join(true_labels)}\nPred: {', '.join(pred_labels)}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

class MultiLabelModel(pl.LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.save_hyperparameters()
    self.cfg = cfg
    self.model = build_cnn(
        num_classes = cfg["model"]["num_classes"],
        backbone = cfg["model"]["backbone"]
    )
    self.lr        = cfg["optimizer"]["lr"]
    self.criterion = nn.BCEWithLogitsLoss()
    self.f1 = MultilabelF1Score(num_labels=cfg["model"]["num_classes"])
    self.auroc = MultilabelAUROC(num_labels=cfg["model"]["num_classes"])

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    images, labels = batch
    logits = self(images)
    loss = self.criterion(logits, labels.float())
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    images, labels = batch
    logits = self(images)
    loss = self.criterion(logits, labels.float())
    #Added the probs since metrics needs this
    probs  = torch.sigmoid(logits)
    self.log("val_loss", loss, prog_bar=True)
    self.log("val_f1",   self.f1(probs, labels.int()),    prog_bar=True)
    self.log("val_auroc",self.auroc(probs, labels.int()), prog_bar=True)
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

  def configure_optimizers(self):
    lr = self.lr if hasattr(self, "lr") else self.cfg["optimizer"]["lr"]
    return torch.optim.Adam(self.parameters(), lr = lr)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if method == "latent":
        encoder, feature_dim = create_encoder(device)

        data_set = load_train_dataset(
            data_dir="aggregated-2",
            batch_size=16,
            num_workers=1,
            image_size=128,
        )

        data_loader = load_train_loader(data_set,
            batch_size=16,
            num_workers=1,
            image_size=128,
        )

        X, Y = extract_features(encoder, data_loader, device)

        print("Feature dim:", feature_dim)
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)

        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=0.15, random_state=42
        )

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val, Y_train_val, test_size=0.1765, random_state=42
        )

        print("X_train shape:", X_train.shape)
        print("Y_train shape:", Y_train.shape)
        print("X_val shape:", X_val.shape)
        print("Y_val shape:", Y_val.shape)
        print("X_test shape:", X_test.shape)
        print("Y_test shape:", Y_test.shape)

        best_pca, best_classifier, best_thresholds, best_result, results = tune_pca_and_classifier(
        X_train, Y_train, X_val, Y_val, clf="logistic"
    )

        plot_tuning_results(results, metric_name="f1_micro", save_path="tuning_results_f1.png")

        plot_tuning_results(results, metric_name="hamming_acc", save_path="tuning_results_hamming.png")


        evaluate_on_test(
        pca=best_pca,
        classifier=best_classifier,
        thresholds=best_thresholds,
        X_test=X_test,
        Y_test=Y_test,
    )


        save_model_bundle(
        output_path="vgg_pca_logreg.pkl",
        pca=best_pca,
        classifier=best_classifier,
        thresholds=best_thresholds,
    )

        print("\nSaved tuned model to vgg_pca_logreg.pkl")
        print("Training complete.")
    else:
        data_set = load_train_dataset(
            data_dir="aggregated-2",
            batch_size=16,
            num_workers=1,
            image_size=128,
        )

        RESNET_CFG = "/configs/resnet_config.yaml"
        EFFICIENTNET_CFG = "/configs/efficientnet_config.yaml"
        DENSENET_CFG = "/configs/densenet121_config.yaml"

        CFG = DENSENET_CFG

        with open(CFG) as f:
            cfg = yaml.safe_load(f)

        checkpoint_callback = ModelCheckpoint(
            dirpath="/content/drive/MyDrive/MLE/densenet/checkpoints/",
            filename="{epoch}-{val_loss:.2f}-{val_f1:.2f}",
            monitor="val_f1",          # save based on best val F1
            mode="max",                # higher F1 is better
            save_top_k=3,              # keep top 3 checkpoints
            save_last=True,            # always save the latest epoch too
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")  # logs LR to wandb

        trainer = pl.Trainer(
        max_epochs=cfg["training"]["num_epochs"],
            check_val_every_n_epoch=cfg["training"]["val_every"],
            accelerator="auto",
            devices="auto",
            log_every_n_steps=10,
            callbacks = [checkpoint_callback, lr_monitor],
        )
        pl_model = MultiLabelModel(cfg)
        tuner = Tuner(trainer)

        data_loader = load_train_dataset(
            data_dir="aggregated-2",
            batch_size=16,
            num_workers=1,
            image_size=128,
        )

        val_size = int(0.2 * len(data_set))
        train_size = len(data_set) - val_size
        train_set, val_set = torch.utils.data.random_split(
            data_set,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2)

        tuner.lr_find(pl_model, train_loader, val_loader)
        trainer.fit(pl_model, train_loader, val_loader)
        print(f"Best model: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
