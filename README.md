# Breast Cancer Diagnosis Prediction

This project uses machine learning techniques to predict breast cancer diagnosis (benign or malignant) based on cell nuclei features. The workflow includes data preprocessing, clustering (K-Means), and classification (K-Nearest Neighbors), with performance evaluation and visualization.

## Features

- **Data Cleaning:** Removes unnecessary columns and encodes diagnosis labels.
- **Normalization:** Scales features using Min-Max normalization.
- **Clustering:** Applies K-Means to group data and evaluates cluster purity.
- **Classification:** Uses K-Nearest Neighbors (KNN) for diagnosis prediction.
- **Evaluation:** Reports accuracy, precision, recall, F1-score, confusion matrix, and optimal K value.
- **Visualization:** Plots clusters, actual diagnoses, confusion matrix, and KNN accuracy vs. K.

## Goals:
- Convert diagnosis labels into numeric encoding: M → 1, B → 0
- Drop non-informative columns (`id` and `Unnamed: 32`)
- Scale features using Min–Max normalization
- Split data into 80% training and 20% testing (stratified)
- Explore clustering structure using K-Means (k = 2)
- Train a KNN classifier with k = 5 and evaluate it

Dataset
-------
The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset (commonly available). It contains 569 instances and 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast mass cell nuclei.

Key dataset stats (as used in this notebook):
- Total samples: 569
- Features: 30 (per-sample)
- Benign cases: 357
- Malignant cases: 212

Column examples:
- id, diagnosis (M/B), radius_mean, texture_mean, perimeter_mean, area_mean, ... , fractal_dimension_worst

Preprocessing
-------------
Primary preprocessing steps performed in the notebook:

1. Read data:
   - Load CSV (e.g., `Dataset.csv`) into a pandas DataFrame.

2. Drop unneeded columns:
   - Drop `id` (not useful as a predictive feature).
   - If present, drop `Unnamed: 32` (often an artifact column).

3. Encode diagnosis:
   - Convert diagnosis from categorical to numeric:
     - 'M' → 1 (malignant)
     - 'B' → 0 (benign)

4. Separate features and target:
   - X: All columns except `diagnosis`
   - y: `diagnosis`

Example code (from the notebook):
```python
columns_to_drop = ['id']
if 'Unnamed: 32' in df.columns:
    columns_to_drop.append('Unnamed: 32')

df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
df_cleaned['diagnosis'] = df_cleaned['diagnosis'].map({'M': 1, 'B': 0})
```

Feature scaling (Min–Max Normalization)
---------------------------------------
To bring features into the same range (0–1), Min–Max normalization is applied to all feature columns (exclude `diagnosis`).

Example:
```python
from sklearn.preprocessing import MinMaxScaler

X = df_cleaned.drop('diagnosis', axis=1)
y = df_cleaned['diagnosis']

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

Train / Test split
------------------
Split data into training and testing sets:
- Train: 80% (used to train the KNN and compute k-selection results)
- Test: 20% (held-out, used for final evaluation)
- Use stratified sampling to preserve class proportions

Example:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)
```

Clustering with K-Means (k = 2)
-------------------------------
K-Means with k = 2 is used to group the normalized feature vectors into two clusters. This is an unsupervised step that helps inspect whether malignant and benign tumors form distinguishable clusters.

- Fit KMeans on the normalized features.
- Compare cluster labels with true diagnosis labels using a cross-tabulation to estimate cluster "purity".

Example:
```python
from sklearn.cluster import KMeans
import pandas as pd

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_normalized)

cross_tab = pd.crosstab(df_cleaned['diagnosis'], cluster_labels)
purity = (cross_tab[0].max() + cross_tab[1].max()) / len(df_cleaned)
print("Cross-tabulation:\n", cross_tab)
print(f"K-Means Purity: {purity:.4f}")
```

Interpretation:
- High purity (close to 1) means clusters largely correspond to true classes.
- In the notebook example, cluster purity ~ 0.9279 (92.79%), indicating strong grouping by diagnosis.

Classification with K-Nearest Neighbors (k = 5)
-----------------------------------------------
A KNN classifier with k = 5 is trained using the training set and evaluated on the held-out test set.

Example:
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

Model selection (optional, included in the notebook):
- The notebook also evaluates accuracies across k values from 1 to 20 and finds the best k (which is k = 5 in the published results).

Evaluation metrics & results
----------------------------
Metrics computed on the test set:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Example:
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
```

## Accuracy & Summary

| Metric             | Value       |
| ------------------ | ----------- |
| Dataset Size       | 569 samples |
| Features           | 30          |
| Benign Cases       | 357         |
| Malignant Cases    | 212         |
| K-Means Purity     | 0.9279      |
| KNN Accuracy (k=5) | 0.9649      |
| Best k             | 5           |
| Best KNN Accuracy  | 0.9649      |

## Usage

### 1. Upload Data

- The notebook expects a CSV file named `dataset.csv` with breast cancer features and diagnosis labels.
- If running locally, ensure `dataset.csv` is in the same directory as the notebook.
- If using Google Colab, upload the file when prompted in the notebook.

### 2. Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Breast_Cancer_Diagnosis_Prediction.ipynb` and `dataset.csv`.
3. Run all cells in order:
   - The notebook will prompt you to upload the dataset if not found.
   - All required libraries are pre-installed in Colab.

### 3. Run Locally

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Place `Breast_Cancer_Diagnosis_Prediction.ipynb` and `dataset.csv` in the same folder.
3. Open the notebook in Jupyter and run all cells.

### 4. Results

- View clustering purity, KNN performance metrics (including accuracy), and visualizations.

## File Structure

- `Breast_Cancer_Diagnosis_Prediction.ipynb` — Main Jupyter notebook for the workflow.
- `dataset.csv` — Input dataset file.

## Notes

- The notebook is compatible with Google Colab (file upload included).
- Adjust the input file name if needed after upload.

