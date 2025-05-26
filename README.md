Markdown

# Machine Failure Prediction System

## Overview
This project implements a machine learning system designed to predict potential equipment failures. By analyzing various sensor readings and operational parameters from industrial machinery, the system identifies patterns indicative of impending failures. The goal is to enable proactive maintenance, thereby minimizing costly downtime, optimizing resource allocation, and extending the lifespan of critical assets.

---

## Project Structure
The repository is organized following best practices for MLOps (Machine Learning Operations) and modular software development. This structure promotes scalability, maintainability, and reproducibility, making it easy for both individual developers and teams to work on the project.

.
├── config.yml
├── data
│   ├── processed
│   └── raw
├── main.py
├── models
│   └── trained_models
├── notebooks
├── requirements.txt
├── results
│   ├── plots
│   └── reports
├── setup.py
├── src
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── utils.py
│   └── visualization.py
└── tests
## Features
-   **End-to-End Pipeline:** Automates the entire machine learning workflow from data ingestion to prediction.
-   **Data Management:** Structured handling of raw and processed data for clear separation and reproducibility.
-   **Configurable Workflow:** Easily adjust data paths, model hyperparameters, and other system settings via `config.yml`.
-   **Modular Design:** Code organized into distinct modules (`src/`) for improved readability, maintainability, and reusability.
-   **Model Training & Persistence:** Supports training various machine learning models (e.g., **[RandomForestClassifier, XGBoostClassifier, LogisticRegression]**) and saving them for later inference.
-   **Comprehensive Evaluation:** Calculates a wide array of performance metrics (Accuracy, ROC-AUC, F1-Score, Precision, Recall) and generates insightful plots (Confusion Matrices, ROC Curves, Model Comparison Charts).
-   **Prediction Capabilities:** Includes functions to load trained models and make predictions on new, unseen operational data.
-   **Version Control Ready:** Integrated with Git for robust version control and collaborative development.

---

## Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:
-   **Python 3.8+**
-   **Git**

### Installation
1.  **Clone the repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone [https://github.com/96sanjay/machine-failure-prediction.git](https://github.com/96sanjay/machine-failure-prediction.git)
    cd machine-failure-prediction
    ```
    (Note: Replace `https://github.com/96sanjay/machine-failure-prediction.git` with your specific repository URL if you change the name or host.)

2.  **Create and activate a virtual environment (highly recommended):**
    A virtual environment isolates your project's dependencies, preventing conflicts with other Python projects.
    ```bash
    python -m venv ml
    # On macOS / Linux:
    source ml/bin/activate
    # On Windows:
    .\ml\Scripts\activate
    ```

3.  **Install project dependencies:**
    Once your virtual environment is active, install all required Python packages:
    ```bash
    pip install -r requirements.txt
    # If you are actively developing the project, you might also want to install it in editable mode:
    # pip install -e .
    ```

### Configuration
-   **Edit `config.yml`:** Open the `config.yml` file in your project's root directory. This file centralizes all configurable parameters, including paths to your raw data, processed data, trained model outputs, and specific hyperparameters for your models. Adjust these settings as needed for your environment and data.
-   **Place Raw Data:** Ensure your raw machine failure datasets (e.g., CSV files) are placed in the `data/raw/` directory, conforming to the paths specified in your `config.yml`.

### Running the Project
To execute the entire machine learning pipeline, from data preprocessing through model training, evaluation, and prediction:

```bash
python main.py
