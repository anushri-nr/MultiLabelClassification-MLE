import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

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

    train_loader = DataLoader(
        train_dataset,
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


def tune_pca_and_classifier(X_train, Y_train, X_val, Y_val, clf):
    candidate_components = [64, 128, 256, 512]
    candidate_thresholds = [0.3, 0.4, 0.5, 0.6]

    results = []
    best_result = None
    best_pca = None
    best_classifier = None

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

        for threshold in candidate_thresholds:
            val_preds = (val_probs >= threshold).astype(np.float32)
            metrics = compute_multilabel_metrics(Y_val, val_preds)

            row = {
                "n_components": n_components,
                "threshold": threshold,
                **metrics,
            }
            results.append(row)

            print(
                f"threshold={threshold:.1f} | "
                f"f1_micro={metrics['f1_micro']:.4f} | "
                f"mean_iou={metrics['mean_iou']:.4f} | "
                f"exact_match={metrics['exact_match']:.4f}"
            )

            if best_result is None or metrics["f1_micro"] > best_result["f1_micro"]:
                best_result = row
                best_pca = pca
                best_classifier = classifier

    print("\nBest validation result:")
    print(best_result)

    return best_pca, best_classifier, best_result, results


def evaluate_on_test(pca, classifier, threshold, X_test, Y_test):
    X_test_pca = pca.transform(X_test)
    test_probs = classifier.predict_proba(X_test_pca)
    test_preds = (test_probs >= threshold).astype(np.float32)

    metrics = compute_multilabel_metrics(Y_test, test_preds)

    print("\nHeld-out test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def save_model_bundle(output_path, pca, classifier, threshold=0.5):
    bundle = {
        "encoder_name": "vgg16",
        "encoder_state_dict": None,
        "pca": pca,
        "classifier": classifier,
        "threshold": threshold,
    }

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)

# def plot_tuning_results(results, save_path="tuning_results.png"):

def plot_tuning_results(results, metric_name="f1_micro", save_path="tuning_results.png"):
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["threshold", "n_components"])

    plt.figure(figsize=(9, 5.5))

    for threshold in sorted(results_df["threshold"].unique()):
        subset = results_df[results_df["threshold"] == threshold]
        plt.plot(
            subset["n_components"],
            subset[metric_name],
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"Threshold = {threshold}"
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
        f"components={int(best_row['n_components'])}, threshold={best_row['threshold']}",
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
    plt.legend(frameon=True)

    plt.subplots_adjust(bottom=0.2)
    plt.figtext(
        0.5,
        0.02,
        "Validation Performance Across PCA Components and Thresholds",
        ha="center",
        fontsize=12
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_sample_predictions(model_bundle, dataset, device, num_samples=6, image_size=128):
    encoder = model_bundle["encoder"]
    pca = model_bundle["pca"]
    classifier = model_bundle["classifier"]
    threshold = model_bundle["threshold"]

    display_transform = transforms.Compose([
        transforms.Resize((image_size, image_size))
    ])

    plt.figure(figsize=(15, 8))

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    for plot_idx, data_idx in enumerate(indices, 1):
        img_path, true_target = dataset.samples[data_idx]

        image = Image.open(img_path).convert("RGB")
        image_display = display_transform(image)

        input_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        x = input_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(x).cpu().numpy()

        features_pca = pca.transform(features)

        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(features_pca)[0]
        else:
            scores = classifier.decision_function(features_pca)[0]
            probs = 1.0 / (1.0 + np.exp(-scores))

        pred_target = (probs >= threshold).astype(np.float32)

        true_labels = [LABEL_ORDER[i] for i, v in enumerate(true_target.tolist()) if v == 1]
        pred_labels = [LABEL_ORDER[i] for i, v in enumerate(pred_target.tolist()) if v == 1]

        plt.subplot(2, 3, plot_idx)
        plt.imshow(image_display)
        plt.axis("off")
        plt.title(
            f"True: {', '.join(true_labels)}\nPred: {', '.join(pred_labels)}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


def visualize_test_predictions(model_bundle, dataset, test_indices, X_test, Y_test, device, num_samples=6, image_size=128):
    encoder = model_bundle["encoder"]
    pca = model_bundle["pca"]
    classifier = model_bundle["classifier"]
    threshold = model_bundle["threshold"]

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

        pred_target = (probs >= threshold).astype(np.float32)

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, feature_dim = create_encoder(device)

    data_loader = load_train_dataset(
        data_dir="aggregated-2",
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

    best_pca, best_classifier, best_result, results = tune_pca_and_classifier(
        X_train, Y_train, X_val, Y_val, clf="logistic"
    )

    # best_pca_svm, best_classifier_svm, best_result_svm, results_svm = tune_pca_and_classifier(
    #     X_train, Y_train, X_val, Y_val, clf="svm"
    # )

    plot_tuning_results(results, metric_name="f1_micro", save_path="tuning_results_f1.png")

    plot_tuning_results(results, metric_name="hamming_acc", save_path="tuning_results_hamming.png")


    evaluate_on_test(
        pca=best_pca,
        classifier=best_classifier,
        threshold=best_result["threshold"],
        X_test=X_test,
        Y_test=Y_test,
    )

    save_model_bundle(
        output_path="vgg_pca_logreg.pkl",
        pca=best_pca,
        classifier=best_classifier,
        threshold=best_result["threshold"],
    )

    print("\nSaved tuned model to vgg_pca_logreg.pkl")
    print("Training complete.")


if __name__ == "__main__":
    main()
