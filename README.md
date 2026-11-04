[README.md](https://github.com/user-attachments/files/34381961/README.md)
# ECG-Based Hypoglycemia Detection Using Deep Learning

A deep learning project exploring whether short electrocardiogram (ECG) recordings can be used to classify **hypoglycemic vs. non-hypoglycemic** states. The implementation combines one-dimensional convolutional layers with a bidirectional LSTM to learn patterns in ECG signals.

> **Academic research prototype:** This model is not validated for clinical diagnosis or medical use.

## Project overview

The main challenge is detecting the minority hypoglycemia class from imbalanced ECG data. The notebook covers data exploration, signal segmentation, model training, and classification evaluation, with particular attention to precision, recall, and F1 score.

## Dataset and preprocessing

- **Source file:** `ECG_10s_2500.csv` (not included in this repository).
- **9 patients** and **45,630 labeled 10-second ECG samples**.
- **43,169 non-hypoglycemic** samples and **2,461 hypoglycemic** samples.
- Each original recording contains **2,500 ECG measurements**.
- Metadata fields are removed before model training.
- Samples are split into **70% training, 15% validation, and 15% testing**, with stratification by label.
- Each 10-second recording is divided into **500-point (2-second) windows**. Training uses **50% overlap**, while validation and test windows do not overlap within each recording.

After segmentation, the notebook reports **287,469 training**, **34,220 validation**, and **34,225 test** windows.

## Model architecture

| Stage | Configuration |
| --- | --- |
| Input | 500 ECG values × 1 channel |
| Convolution 1 | Conv1D, 32 filters, kernel size 15, ReLU; max pooling |
| Convolution 2 | Conv1D, 64 filters, kernel size 11, ReLU; max pooling |
| Convolution 3 | Conv1D, 128 filters, kernel size 7, ReLU |
| Sequence modeling | Bidirectional LSTM, 64 units |
| Classification | Dense 64, ReLU; dropout 0.3; sigmoid output |

**Training:** Adam (learning rate `0.001`), binary cross-entropy, up to 40 epochs, batch size 128, and class weights `{0: 1, 1: 4}`. A `ModelCheckpoint` callback monitors validation PR-AUC and saves the best model to `best_model.h5`.

## Evaluation results

The saved notebook output reports the following results for the **hypoglycemia class (1)**:

| Metric | Reported value |
| --- | ---: |
| Precision | 0.7873 |
| Recall / sensitivity | 0.7724 |
| F1 score | 0.7798 |
| Specificity | 0.9881 |
| Overall accuracy | 0.9765 |

Confusion matrix (rows: true label; columns: predicted label):

```text
                 Predicted 0   Predicted 1
Actual 0             31995           385
Actual 1               420          1425
```

**Interpretation note:** These are exploratory results, **not an unbiased final test estimate**. The notebook selects the F1-maximizing classification threshold (`0.7261`) using the *test-set labels*, which introduces evaluation bias. A more rigorous experiment should select the threshold on validation data and evaluate it once on an untouched test set. Also, the current split is stratified by sample rather than by patient; patient-level separation is needed to measure generalization to unseen patients. The checkpoint is saved during training, but the evaluation cell uses the final in-memory model rather than explicitly reloading the best checkpoint.

## Technologies

Python, pandas, NumPy, Matplotlib, Keras, scikit-learn, and Google Colab.

## Running the notebook

1. Open `Code.ipynb` in **Google Colab**.
2. Install the required packages if they are not already available:

   ```bash
   pip install pandas numpy matplotlib tensorflow scikit-learn pydot
   ```

3. Make the dataset available in your Google Drive at `MyDrive/COE 547 - Final Project/ECG_10s_2500.csv`, or update the path in the loading cell.
4. Run the cells in order to inspect the data, create ECG windows, train the network, and view the evaluation output.

The dataset is required to reproduce the experiment and is not included here. Ensure you have permission to use or distribute it. The model diagram step may require Graphviz in the execution environment. Depending on the installed Keras version, the `.h5` checkpoint format may need updating.

## Future improvements

- Use a **patient-wise train/validation/test split** to evaluate generalization to unseen people.
- Choose the F1 threshold using **validation data only**.
- Reload the best saved checkpoint before final evaluation.
- Compare against simpler baselines and report repeated-run variability.
- Investigate class imbalance strategies and error patterns, particularly false negatives.

## Repository contents

- `Code.ipynb`: data preparation, CNN-BiLSTM training, and evaluation.

## Disclaimer

This is an academic machine learning experiment. Its results should not be interpreted as evidence of clinical safety or diagnostic performance.
