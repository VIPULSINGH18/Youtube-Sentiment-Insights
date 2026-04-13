<div align="center">

<img src="https://img.shields.io/badge/YouTube-Sentiment%20Insights-FF0000?style=for-the-badge&logo=youtube&logoColor=white"/>

# 🎯 YouTube Sentiment Insights

### An end-to-end MLOps pipeline that classifies YouTube comments as Positive, Negative, or Neutral — served via a Flask API and a Chrome Extension.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-success?style=flat-square)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.17.0-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.53.0-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [MLOps Pipeline (DVC)](#-mlops-pipeline-dvc)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running the Pipeline](#-running-the-pipeline)
- [Flask API](#-flask-api)
- [Chrome Extension](#-chrome-extension)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [CI/CD](#-cicd)
- [Results](#-results)
- [Author](#-author)

---

## 🔍 Overview

**YouTube Sentiment Insights** is a production-ready ML system built with an end-to-end MLOps mindset. It scrapes YouTube comments, preprocesses them, trains a **LightGBM** classifier with **TF-IDF (trigrams)**, handles class imbalance via **SMOTE**, tunes hyperparameters with **Optuna**, tracks experiments on **MLflow (hosted on AWS EC2)**, versions data with **DVC + S3**, and serves predictions through a **Flask REST API** — accessible directly from a **Chrome Extension** on any YouTube video page.

**Key highlights:**
- 🧪 Full reproducibility via DVC pipeline with locked stages
- 📊 MLflow experiment tracking with S3 artifact storage
- 🐳 Dockerized API for consistent deployments
- 🔁 GitHub Actions CI/CD for automated testing & deployment
- 🧩 Chrome Extension frontend for real-time YouTube integration

---

## 🏗️ Architecture

```
YouTube Comments
      │
      ▼
┌─────────────────┐
│  Data Ingestion │  ← YouTube API / CSV
│   (DVC Stage)   │
└────────┬────────┘
         │ data/raw
         ▼
┌──────────────────────┐
│  Data Preprocessing  │  ← NLTK cleaning, TF-IDF, SMOTE
│     (DVC Stage)      │
└────────┬─────────────┘
         │ data/interim
         ▼
┌─────────────────────┐
│   Model Building    │  ← LightGBM + Optuna tuning
│    (DVC Stage)      │
└────────┬────────────┘
         │ lgbm_model.pkl + tfidf_vectorizer.pkl
         ▼
┌──────────────────────┐
│  Model Evaluation   │  ← MLflow logging → AWS EC2
│    (DVC Stage)       │       Artifacts → S3
└────────┬─────────────┘
         │ experiment_info.json
         ▼
┌───────────────────────┐
│  Model Registration  │  ← MLflow Model Registry
│    (DVC Stage)        │
└────────┬──────────────┘
         │
         ▼
┌──────────────────┐      ┌──────────────────────────┐
│   Flask REST API │ ←──→ │  Chrome Extension (UI)   │
│   (Dockerized)   │      │  Real-time on YouTube     │
└──────────────────┘      └──────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tool / Library |
|---|---|
| **ML Model** | LightGBM 4.5.0 |
| **Text Features** | TF-IDF (trigrams, scikit-learn) |
| **Imbalance Handling** | SMOTE (imbalanced-learn) |
| **Hyperparameter Tuning** | Optuna |
| **Experiment Tracking** | MLflow 2.17.0 |
| **Data Versioning** | DVC 3.53.0 + DVC[S3] |
| **Cloud Infrastructure** | AWS EC2 (MLflow server) + S3 (artifacts) |
| **Backend API** | Flask 3.0.3 + Flask-CORS |
| **Frontend** | Chrome Extension (Vanilla JS) |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Text Preprocessing** | NLTK 3.9.1 |
| **AWS SDK** | boto3 1.35.36 |
| **Visualization** | Seaborn, Matplotlib, WordCloud |

---

## 🔁 MLOps Pipeline (DVC)

The entire ML workflow is defined as a **reproducible DVC pipeline** with 5 stages:

```
data_ingestion → data_preprocessing → model_building → model_evaluation → model_registration
```

### Stage Details

| Stage | Script | Input | Output |
|---|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | params: `test_size` | `data/raw/` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | `data/raw/train.csv`, `test.csv` | `data/interim/` |
| `model_building` | `src/model/model_building.py` | `data/interim/train_processed.csv` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | model + processed data | `experiment_info.json` |
| `model_registration` | `src/model/register_model.py` | `experiment_info.json` | MLflow Model Registry |

### Pipeline Parameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.2

model_building:
  max_features: 10000
  ngram_range: [1, 3]      # trigrams
  learning_rate: 0.05
  max_depth: 7
  n_estimators: 300
```

---

## 📁 Project Structure

```
Youtube-Sentiment-Insights/
│
├── .dvc/                          # DVC configuration
├── .github/workflows/             # GitHub Actions CI/CD
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py      # Fetch & split raw data
│   │   └── data_preprocessing.py # Clean, TF-IDF, SMOTE
│   └── model/
│       ├── model_building.py      # LightGBM + Optuna training
│       ├── model_evaluation.py    # Metrics + MLflow logging
│       └── register_model.py     # MLflow model registration
│
├── flask_app/                     # REST API
├── notebooks/                     # EDA & experiments
├── yt-chrome-plugin-frontend/     # Chrome Extension
│
├── dvc.yaml                       # Pipeline definition
├── dvc.lock                       # Reproducibility lock
├── params.yaml                    # Hyperparameters
├── Dockerfile                     # Container setup
├── requirements.txt
└── setup.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker
- AWS credentials (EC2 + S3 access)
- MLflow tracking server running on EC2

### 1. Clone the Repository

```bash
git clone https://github.com/VIPULSINGH18/Youtube-Sentiment-Insights.git
cd Youtube-Sentiment-Insights
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export MLFLOW_TRACKING_URI=http://<your-ec2-public-ip>:5000
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
```

### 4. Pull Data with DVC

```bash
dvc pull
```

---

## ▶️ Running the Pipeline

Run the full end-to-end pipeline with a single command:

```bash
dvc repro
```

To run a specific stage:

```bash
dvc repro model_building
```

To visualize the DAG:

```bash
dvc dag
```

---

## 🌐 Flask API

### Run Locally

```bash
cd flask_app
python app.py
```

### Run with Docker

```bash
docker build -t yt-sentiment-api .
docker run -p 5000:5000 yt-sentiment-api
```

### API Endpoint

```http
POST /predict
Content-Type: application/json

{
  "comment": "This video is absolutely amazing, loved every second!"
}
```

**Response:**

```json
{
  "sentiment": "Positive",
  "confidence": 0.94
}
```

---

## 🧩 Chrome Extension

The `yt-chrome-plugin-frontend/` directory contains a browser extension that integrates directly with YouTube.

**How to load it:**
1. Open Chrome → `chrome://extensions/`
2. Enable **Developer Mode**
3. Click **Load Unpacked** → select `yt-chrome-plugin-frontend/`
4. Navigate to any YouTube video and click the extension icon

The extension fetches comments from the current video, sends them to the Flask API, and displays a sentiment breakdown in real time.

---

## 📊 MLflow Experiment Tracking

All training runs are logged to an **MLflow Tracking Server hosted on AWS EC2**, with artifacts stored in **S3**.

**Tracked per experiment:**
- Hyperparameters (learning rate, max depth, n_estimators, etc.)
- Metrics: Accuracy, F1-Score, Precision, Recall
- Artifacts: trained model, TF-IDF vectorizer, confusion matrix
- Data version hash (via DVC)

**Access the UI:**
```
http://ec2-13-201-86-39.ap-south-1.compute.amazonaws.com:5000/
```

---

## 🔁 CI/CD

GitHub Actions is configured to automatically:

- ✅ Run tests on every push to `main`
- 🐳 Build and push the Docker image
- 🚀 Deploy the updated API to the target environment

Pipeline config: `.github/workflows/`

---

## 📈 Results

| Metric | Score |
|---|---|
| **F1-Score(Weighted)** | ~78.16% |
| **F1-Score (Macro)** | ~76.56% |
| **Model** | LightGBM |
| **Features** | TF-IDF Trigrams |
| **Classes** | Positive / Negative / Neutral |

> Confusion matrix is available in the repo root: `confusion_matrix_Test Data.png`

---

## 👤 Author

**Vipul Singh**
AI/ML Engineer | GenAI Enthusiast | MLOps Builder

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vipulsk04-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/vipulsk04)
[![GitHub](https://img.shields.io/badge/GitHub-VIPULSINGH18-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/VIPULSINGH18)

---

<div align="center">

⭐ **If this project helped you, drop a star!** ⭐

</div>
