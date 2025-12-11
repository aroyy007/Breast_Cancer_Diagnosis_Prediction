# Breast Cancer Diagnosis Prediction

This project uses machine learning techniques to predict breast cancer diagnosis (benign or malignant) based on cell nuclei features. The workflow includes data preprocessing, clustering (K-Means), and classification (K-Nearest Neighbors), with performance evaluation and visualization.

## Features

- **Data Cleaning:** Removes unnecessary columns and encodes diagnosis labels.
- **Normalization:** Scales features using Min-Max normalization.
- **Clustering:** Applies K-Means to group data and evaluates cluster purity.
- **Classification:** Uses K-Nearest Neighbors (KNN) for diagnosis prediction.
- **Evaluation:** Reports accuracy, precision, recall, F1-score, confusion matrix, and optimal K value.
- **Visualization:** Plots clusters, actual diagnoses, confusion matrix, and KNN accuracy vs. K.

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

