<div align="center">

# Heart Disease Risk Predictor

### KNN classification with an end-to-end ML workflow

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

Predict heart disease risk from clinical measurements using a K-Nearest Neighbors model.

**Live Hosted**: <http://knn-classification-heartdisease.streamlit.app/>

[Overview](#-overview) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Dataset](#-dataset)

</div>

---

## Overview

This project builds a complete ML pipeline for heart disease risk prediction, from data cleaning to model deployment. It includes:

- A **KNN classifier** trained on cleaned UCI heart disease data
- Saved preprocessing and model artifacts for reproducible inference
- A **Streamlit** app for single and batch predictions

### Key Highlights

| Aspect | Details |
|--------|---------|
| **Model** | K-Nearest Neighbors (KNN) Classifier |
| **Target** | Binary risk classification |
| **Tools Used** | Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| **Deployment** | Streamlit Web App |
| **Artifacts** | `knn_pipeline.joblib`, `preprocessor.joblib` |

---

## Features

- **Single Prediction**: Enter patient data and get instant risk classification
- **Batch Predictions**: Upload CSV/TXT/TSV/Excel/JSON and download results
- **Reusable Pipeline**: Saved preprocessing + model pipeline
- **Reproducible Workflow**: Step-by-step notebooks for the full ML process

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "KNN Classification"
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
   ```

---

## Usage

### Running the Streamlit App

```bash
streamlit run app/app.py
```

The app opens in your browser at `http://localhost:8501`.

### Making Predictions

1. Enter patient details in the form
2. Click **Predict Risk**
3. Review the predicted risk category and probability (if available)

### Exploring the Notebooks

```bash
jupyter notebook notebooks/
```

**Available Notebooks:**

- `01_data_exploration.ipynb` - Initial data exploration
- `02_data_cleaning.ipynb` - Data cleaning and preprocessing
- `03_eda.ipynb` - Exploratory data analysis
- `04_model_building.ipynb` - Model training and evaluation
- `05_KNN_Classification.ipynb` - Final KNN pipeline and testing

---

## Project Structure

```
KNN Classification/
│
├── README.md
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
│       ├── cleaned_heart.csv
│       ├── X_test.joblib
│       ├── X_train.joblib
│       ├── y_test.joblib
│       └── y_train.joblib
│
├── models/
│   ├── knn_pipeline.joblib
│   └── preprocessor.joblib
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_data_cleaning.ipynb
    ├── 03_eda.ipynb
    ├── 04_model_building.ipynb
    └── 05_KNN_Classification.ipynb
```

---

## Dataset

**Source**: UCI Heart Disease Dataset (Cleveland subset)  
**Location**: `data/raw/heart.csv`

### Description

The dataset contains patient-level clinical features used to predict heart disease risk.

**Example Features:**

- Age, sex, chest pain type
- Resting blood pressure, serum cholesterol
- Resting ECG, max heart rate, exercise-induced angina
- ST depression, slope, number of vessels, thalassemia

---

## Technologies Used

- **Python 3.8+**
- **Streamlit**
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **Joblib**

---

## Future Enhancements

- [ ] Evaluate additional classifiers (Random Forest, XGBoost)
- [ ] Add model explainability with SHAP/LIME
- [ ] Improve data validation for uploaded batch files
- [ ] Add a REST API for predictions

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## License

Open Source Project - Feel free to use and modify as needed.

---

## Author

**Meet Bataviya**

Created as a Machine Learning project for heart disease risk classification.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/meet-bataviya/)
---

If you found this helpful, please star the repository.
