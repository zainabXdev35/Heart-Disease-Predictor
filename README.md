# Heart Disease Prediction System

A machine-learning web app that predicts whether a patient is likely to have heart disease from **four** clinical inputs. Built with Python, scikit-learn and Streamlit.

> **Live demo:** _paste your Streamlit URL here after deployment_ (`https://<your-app>.streamlit.app`)

> **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for diagnosis. Always consult a doctor.

---

## Table of Contents

1. [What the model does](#1-what-the-model-does)
2. [Inputs and output](#2-inputs-and-output)
3. [Dataset](#3-dataset)
4. [How it works (pipeline)](#4-how-it-works-pipeline)
5. [Model details](#5-model-details)
6. [Performance](#6-performance)
7. [Repository structure](#7-repository-structure)
8. [Run locally](#8-run-locally)
9. [Deploy to Streamlit Cloud](#9-deploy-to-streamlit-cloud)
10. [Retrain the model](#10-retrain-the-model)
11. [Limitations](#11-limitations)

---

## 1. What the model does

The system performs **binary classification**: given a patient's measurements, it answers one question, _"Is heart disease likely: Yes or No?"_

It uses a **Support Vector Classifier (SVC)** that was trained on 303 patient records. During training the model learned a boundary that separates patients with heart disease from patients without it. For a new patient, it checks which side of that boundary the patient falls on.

## 2. Inputs and output

Only 4 of the 13 original attributes are used.

| Input | Original column | Type | Allowed values |
|---|---|---|---|
| Chest pain type | `cp` | Categorical | Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic |
| Maximum heart rate achieved | `thalach` | Numeric | about 70 to 210 bpm |
| Exercise-induced angina | `exang` | Categorical | Yes, No |
| ST depression induced by exercise | `oldpeak` | Numeric | 0.0 to 6.5 |

| Output | Values |
|---|---|
| Heart disease | **Yes** (likely) or **No** (not detected) |

**Why these four?** They rank among the strongest predictors (correlation, Random Forest importance, cross-validated accuracy) and all come from a symptom review plus a basic exercise stress test, so a user can realistically enter them. Stronger attributes such as `ca` and `thal` were excluded because they need a thallium scan and fluoroscopy.

## 3. Dataset

- **Source:** UCI Machine Learning Repository, Heart Disease dataset (Cleveland file `processed.cleveland.data`), https://archive.ics.uci.edu/dataset/45/heart+disease
- **Size:** 303 patients, 14 columns (13 attributes + 1 target)
- **Target:** `num` (0 to 4) converted to binary: 0 = **No**, 1 to 4 = **Yes**
- **Class balance:** 164 No (54%) and 139 Yes (46%)
- **Missing values:** 6 rows (in `ca` and `thal`), none in the 4 selected inputs, so no rows were removed

## 4. How it works (pipeline)

Every new patient passes through the same steps that were applied to the training data:

```
User input (4 values)
      |
      v
1. Build feature vector     one-row table, same column names and order as training
      |
      v
2. Label encoding           label_encoders.pkl   text -> numbers
      |                     (ChestPain, ExerciseAngina)
      v
3. Scaling                  scaler.pkl           (x - mean) / std for all 4 columns
      |
      v
4. SVC prediction           svc_trained_model.pkl  -> 0 or 1
      |
      v
Result: "HEART DISEASE LIKELY" / "NO HEART DISEASE DETECTED"
```

**Encoding mapping** (alphabetical, learned by `LabelEncoder`):

| ChestPain | Code |
|---|---|
| Asymptomatic | 0 |
| Atypical Angina | 1 |
| Non-anginal Pain | 2 |
| Typical Angina | 3 |

ExerciseAngina: No = 0, Yes = 1. HeartDisease: No = 0, Yes = 1.

**Why scaling?** SVC works on distances, so a large-valued column (heart rate around 150) would otherwise dominate 0/1 columns. Scaling raised test accuracy from 0.64 to 0.80. The scaler is fitted on **training data only** to avoid data leakage; the app only calls `transform`, never `fit`.

## 5. Model details

| Item | Value |
|---|---|
| Algorithm | Support Vector Classifier (`sklearn.svm.SVC`) |
| Kernel | RBF (default) |
| `gamma` | `'auto'` (= 1 / 4 features = 0.25) |
| `C` | 1.0 (default) |
| Train / test split | 80% / 20% (242 / 61 patients), stratified, `random_state=0` |
| Support vectors | 133 of 242 training patients |

**Models compared** on the same data: SVC, Logistic Regression, Decision Tree.

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| SVC (deployed) | 0.80 | 0.79 |
| Logistic Regression | 0.89 | 0.79 |
| Decision Tree | 0.69 | 0.69 |

Logistic Regression's higher test score is mostly luck of a small 61-row test set; in cross-validation it ties with SVC. SVC is deployed to follow the reference (Titanic) project.

## 6. Performance

Evaluated on 61 unseen test patients:

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 32 | 1 |
| **Actual: Yes** | 11 | 17 |

| Metric | Value |
|---|---|
| Accuracy | 0.80 |
| Precision (disease) | 0.94 |
| Recall (disease) | 0.61 |
| Baseline (always answer "No") | 0.54 |

When the model says "disease" it is almost always right, but it **misses about 4 in 10 sick patients**.

## 7. Repository structure

```
heart-disease-app/
|-- app.py                   Streamlit web app (input form + prediction)
|-- requirements.txt         Python packages for deployment
|-- svc_trained_model.pkl    Trained SVC model
|-- scaler.pkl               Fitted StandardScaler
|-- label_encoders.pkl       Fitted LabelEncoders (ChestPain, ExerciseAngina, HeartDisease)
`-- README.md                This file
```

All files must stay in the **same folder**, because `app.py` loads the `.pkl` files from its own directory.

## 8. Run locally

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 9. Deploy to Streamlit Cloud

1. Create a **public** GitHub repository (e.g. `heart-disease-app`).
2. Upload `app.py`, `requirements.txt`, `svc_trained_model.pkl`, `scaler.pkl`, `label_encoders.pkl`.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** -> choose the repository, branch `main`, main file `app.py`.
5. Open **Advanced settings**, select **Python 3.12**, then click **Deploy**.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` / install fails | Set Python to 3.11 or 3.12 in app Settings |
| Error or warning loading a `.pkl` | `scikit-learn` in `requirements.txt` must match the version that created the files (1.8.0). Check yours with `pip show scikit-learn` |
| `requirements.txt` ignored | It must be in the same folder as `app.py` |

## 10. Retrain the model

The full workflow (loading, preprocessing, feature selection, encoding, training, testing, application, feedback, deployment) is in `Heart_Disease_Prediction_System.ipynb`.

1. Place `processed_cleveland.data` next to the notebook.
2. Run all cells (**Kernel -> Restart & Run All**). This regenerates the three `.pkl` files.
3. Copy the new `.pkl` files into this folder and pin your `scikit-learn` version in `requirements.txt`.

## 11. Limitations

- **Small data:** 303 patients from one hospital; results are noisy and may not generalise.
- **Only 4 inputs:** limits achievable recall; adding `ca`, `thal`, `age` improves accuracy (about 0.825 in cross-validation) but needs clinical tests.
- **Missed cases:** recall for disease is 0.61, so a "No" result does not rule out heart disease.
- **No tuning:** default `C`, and `gamma='auto'`.
- **Label encoding** gives chest-pain categories an artificial order (no measurable accuracy cost here).
- **Yes/No only:** no probability or confidence score is shown.
- **Not clinically validated.**

## License and credits

Dataset: UCI Machine Learning Repository (Janosi, Steinbrunn, Pfisterer, Detrano). Educational use only.
