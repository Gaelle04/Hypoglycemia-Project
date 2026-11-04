# Hypoglycemia Detection from ECG Signals Using Deep Learning

A deep learning project exploring whether electrocardiogram (ECG) signals can be used to classify hypoglycemic and non-hypoglycemic recordings. The project combines **1D convolutional neural networks (CNNs)** and **bidirectional long short-term memory networks (BiLSTMs)** to learn patterns from ECG time series.

> **Academic research prototype:** This model is an experimental classifier, not a validated medical diagnostic tool.

## Project overview

Hypoglycemia is a condition in which blood glucose levels become abnormally low. This project investigates ECG-based classification as a potential non-invasive approach to recognizing patterns associated with hypoglycemia.

The main technical challenges include **strong class imbalance**, processing sequential ECG data, and assessing performance beyond accuracy alone.

## Dataset

The notebook loads `ECG_10s_2500.csv` from Google Drive. The dataset itself is **not included** in this repository.

| Property | Value |
| --- | --- |
| Patients | 9 |
| Original ECG recordings | 45,630 |
| Non-hypoglycemic recordings (`0`) | 43,169 |
| Hypoglycemic recordings (`1`) | 2,461 |
| Original signal length | 2,500 samples per recording |
| Model input | 500-sample windows (2 seconds, based on the notebook's window annotation) |

The positive class represents approximately **5.4%** of the original recordings.

## Methodology

### 1. Data preparation

- Load the ECG dataset and inspect the label distribution by patient.
- Remove metadata columns (`day`, `count`, `glucose_values`, and `patient_id`) before constructing the model inputs.
- Split recordings into training (70%), validation (15%), and test (15%) sets using stratified random splitting.
- Divide the original recordings into 500-sample windows. The notebook uses **75% overlap for training** and **no overlap for validation or test**.
- Explore random undersampling of the majority class as a class-balancing strategy.

**Implementation note:** Although the notebook creates an undersampled training set, the model's `fit()` call uses the full windowed training set with class weights instead. The undersampled data is not used for the displayed training run.

### 2. Model architecture

The classifier uses three convolutional blocks to extract ECG features, followed by two bidirectional LSTM layers to model temporal patterns.

```text
Input ECG window (500 × 1)
        ↓
Conv1D (32 filters, kernel 15) → MaxPooling → BatchNorm
        ↓
Conv1D (64 filters, kernel 11) → MaxPooling → BatchNorm
        ↓
Conv1D (128 filters, kernel 7) → MaxPooling → BatchNorm
        ↓
Bidirectional LSTM (64 units)
        ↓
Bidirectional LSTM (32 units)
        ↓
Dropout (0.3)
        ↓
Dense (32, ReLU) → Dropout (0.3)
        ↓
Dense (1, Sigmoid)
        ↓
Hypoglycemia probability
```

**Total parameters:** 223,617.

### 3. Training

| Setting | Notebook configuration |
| --- | --- |
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Binary cross-entropy |
| Batch size | 128 |
| Epochs requested | 40 |
| Class weights | Non-hypoglycemia: 1.0; hypoglycemia: 4.0 |
| Tracked metrics | ROC-AUC, PR-AUC, precision, recall |

### 4. Evaluation

The notebook calculates predicted probabilities and uses a precision-recall curve to choose a classification threshold that maximizes F1 score. It then computes:

- Confusion matrix
- Precision, recall, and F1 score
- Sensitivity and specificity
- Classification report

**Results:** The uploaded notebook does not contain saved final test metrics, so numerical performance claims are intentionally omitted. Run the evaluation cell to generate the results before adding them here.

**Evaluation caveat:** The current notebook chooses the F1-optimal threshold using test labels. For an unbiased assessment, choose this threshold on the **validation set**, then evaluate it once on the **held-out test set**.

## Technologies

Python, NumPy, pandas, Matplotlib, TensorFlow/Keras, scikit-learn, imbalanced-learn, and Google Colab.

## How to run

1. Open `Final-version-DLproject.ipynb` in Google Colab or Jupyter.
2. Install the dependencies:

   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn imbalanced-learn
   ```

3. Obtain the ECG dataset from its authorized source and update the `file_path` variable in the notebook to point to `ECG_10s_2500.csv`.
4. Run the cells in order to preprocess the signals, train the model, and evaluate its predictions.

The notebook currently uses a Google Drive path, so local execution requires changing that path. The dataset is not supplied here, and no public download source is verified by this README.

## Limitations and future improvements

- **Patient-level generalization:** The notebook uses a stratified random recording split rather than a patient-disjoint split. Recordings from the same patient can appear across subsets, so the reported setup does not establish performance on unseen patients.
- **Threshold selection:** Select the classification threshold on validation data instead of test data.
- **Window-level dependence:** Overlapping training windows are highly correlated. Interpret the effective sample size accordingly.
- **Class imbalance:** Compare class weighting with other balancing strategies and report precision-recall metrics.
- **Reproducibility:** Add a dataset citation, environment versions, and final test results when available.

## Project context

Developed as an academic deep learning project for **COE 547** at the **Lebanese American University (LAU)**.

The project strengthened my interest in time-series modeling, imbalanced classification, model evaluation, and building more reliable machine learning systems.
