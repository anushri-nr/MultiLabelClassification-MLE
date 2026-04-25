# Multi-Label Image Classification

This project trains and evaluates models for multi-label image classification, where each image may contain any subset of 12 everyday classroom or desk objects:

`pen`, `paper`, `book`, `clock`, `phone`, `laptop`, `chair`, `desk`, `bottle`, `keychain`, `backpack`, `calculator`

The task is to predict the presence or absence of each object independently. This is different from standard single-label classification because an image can contain multiple valid labels at once, such as `pen_paper` or `book_clock_laptop`.

The project explores two modeling approaches:

1. The main deep transfer-learning approach using ResNet-50, EfficientNet-B3, and DenseNet-121 style CNN backbones.
2. Classical transfer-learning baselines using frozen VGG16 features, PCA, and one-vs-rest logistic regression or SVM classifiers.

The strongest reported model was EfficientNet-B3, which achieved a validation micro-F1 score of `0.8364`, exact-match accuracy of `0.6344`, hamming accuracy of `0.9526`, and mean IoU of `0.7957`.

## Project Structure

```text
.
├── data                         # Training dataset
├── configs/                              # CNN backbone config files
│   ├── densenet121_config.yaml
│   ├── efficientnet_config.yaml
│   └── resnet_config.yaml
├── mle-project-train.py                  # Training script
├── mle-project-eval.py                   # Evaluation script
├── requirements.txt                      # Python dependencies
├── tuning_results_*.png                  # PCA tuning plots
└── vgg_pca_*.pkl                         # Saved VGG + PCA + classifier models
```

## Dataset Format

The dataset is expected to use a custom directory layout where each subdirectory name contains one or more labels separated by underscores. Each image file inside those folders should be named like `img1.png`, `img2.png`, etc.

Example:

```text
data/
├── chair_bottle/
│   ├── img1.png
│   └── img2.png
├── paper_chair_keychain/
│   ├── img1.png
│   └── img2.png
└── paper/
    └── img1.png
```

Only folders whose label names match the 12 supported labels are loaded.

The final project dataset contains `4,543` labeled images. Using an 80/20 split, this gives `3,635` training samples and `908` validation samples.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The VGG16 weights are loaded from `torchvision`, so the first training or evaluation run may download pretrained weights if they are not already cached.

## Training

### Main Approach: Deep Transfer Learning

The primary project approach fine-tunes modern CNN backbones:

- `ResNet-50`: uses residual skip connections to support deeper neural networks.
- `EfficientNet-B3`: uses compound scaling to balance depth, width, and resolution.
- `DenseNet-121`: uses dense connections to encourage feature reuse and improve gradient flow.

The deep models replace the final classifier layer with a 12-output linear layer and use binary cross-entropy with logits for multi-label prediction. These models substantially outperformed the classical baselines in validation performance.

`mle-project-train.py` contains a `--method deep` path for PyTorch Lightning training using the YAML files in `configs/`. The available config files target ResNet50, EfficientNet-B3, and DenseNet121-style training.

### Baselines: VGG16 Features + PCA + Classifier

Run the latent-feature training pipeline:

```bash
python mle-project-train.py --method latent --data_dir data
```

This pipeline:

1. Loads images from the dataset directory.
2. Resizes images to `128 x 128`.
3. Extracts frozen VGG16 features.
4. Splits the data into train, validation, and held-out test sets.
5. Tunes PCA component counts across `64`, `128`, `256`, and `512`.
6. Tunes per-class probability thresholds on the validation split.
7. Evaluates the best model on the held-out test split.
8. Saves the trained model bundle to `vgg_pca_logreg.pkl`.

The script also writes tuning plots such as:

```text
tuning_results_f1.png
tuning_results_hamming.png
```

Existing saved model artifacts in this repository include logistic regression and SVM variants:

```text
vgg_pca_logreg.pkl
vgg_pca_logistic_regression.pkl
vgg_pca_svm.pkl
```

For logistic regression, the best validation configuration used `256` PCA components with a decision threshold of `0.4`. For the calibrated linear SVM, the best configuration used `128` PCA components with a decision threshold of `0.3`.

## Evaluation

Evaluate a saved model on a test dataset with the same folder layout:

```bash
python mle-project-eval.py \
  --model_path vgg_pca_logreg.pkl \
  --test_data path_to_test_data \
  --group_id 8 \
  --project_title "Multi Label Classification"
```

The evaluation script reports:

- Binary cross-entropy loss
- Exact match accuracy
- Hamming accuracy
- Mean IoU / Jaccard score
- Micro precision
- Micro recall
- Micro F1

## Results

Validation performance on the held-out set of 908 images:

| Model | Exact Match | Hamming Acc. | Mean IoU | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.3739 | 0.9086 | 0.6134 | 0.7154 | 0.6465 | 0.6792 |
| SVM | 0.3591 | 0.9063 | 0.5959 | 0.7132 | 0.6253 | 0.6921 |
| DenseNet-121 | 0.6068 | 0.9486 | 0.7763 | 0.8591 | 0.7827 | 0.8191 |
| EfficientNet-B3 | 0.6344 | 0.9526 | 0.7957 | 0.8598 | 0.8142 | 0.8364 |
| ResNet-50 | 0.6244 | 0.9502 | 0.7903 | 0.8402 | 0.8210 | 0.8305 |

The deep transfer-learning models substantially outperformed the classical VGG-feature baselines across every metric. EfficientNet-B3 achieved the best overall performance, while ResNet-50 produced the highest recall. Logistic regression and SVM are included as baseline comparisons.

## Metrics

Because this is a multi-label problem, the project uses several complementary metrics:

- **Exact match accuracy**: the fraction of images where the full predicted label set exactly matches the ground truth.
- **Hamming accuracy**: average per-label correctness across all images and classes.
- **Mean IoU / Jaccard index**: average set overlap between predicted and true labels.
- **Micro precision, recall, and F1**: aggregate binary decisions across all labels before computing precision, recall, and F1.


## Limitations and Future Work

The dataset is moderately sized, and rare label combinations are underrepresented. Future improvements could include:

- Expanding the dataset with more images and more rare label combinations.
- Using focal loss or class-balanced sampling to improve performance on underrepresented labels.
- Improving per-class threshold calibration.
- Moving from image-level classification to object detection with YOLO, DETR, or similar models for localization in cluttered scenes.
- Exploring zero-shot or few-shot learning for unseen object combinations.

## Notes

- All labels are multi-hot encoded in the fixed order listed at the top of this README.
- The default image size is `128`.
- The project expects PNG files with names matching `img*.png`.

## References

- He et al., "Deep Residual Learning for Image Recognition", 2016.
- Huang et al., "Densely Connected Convolutional Networks", 2017.
- Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", 2014.
- Tan and Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", 2019.
- Zhang and Zhou, "A Review on Multi-Label Learning Algorithms", 2014.
