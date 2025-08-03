Machine Learning Assignment 5: Naive Bayes, Decision Trees, and Ensemble Learning

This repository contains Jupyter notebooks and scripts for Tasks 1–9 covering Naive Bayes, Decision Trees, and Ensemble Methods, using text and numeric datasets. Follow the instructions below to reproduce the results.

📂 Directory Structure

├── notebooks/
│   ├── Task1_Theory_NaiveBayes.ipynb
│   ├── Task2_SpamDetection_MultinomialNB.ipynb
│   ├── Task3_GaussianNB_Iris.ipynb
│   ├── Task4_Conceptual_DecisionTree.md
│   ├── Task5_DecisionTree_Titanic.ipynb
│   ├── Task6_ModelTuning_Titanic.ipynb
│   ├── Task7_Conceptual_Ensemble.md
│   ├── Task8_RF_vs_DT.ipynb
│   └── Task9_AdaBoost_Titanic.ipynb
├── data/
│   ├── sms.tsv              # SMS Spam Collection dataset
│   └── train_cleaned.csv    # Cleaned Titanic dataset
├── README.md
└── requirements.txt

 
 📋 Requirements

* Python 3.8+
* pandas
* scikit-learn
* matplotlib

Install dependencies via:
bash
pip install -r requirements.txt
 📥 Data Sources

1. SMS Spam Collection (Task 2)

   * Link: [https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
   * Preprocessed `.tsv` included as `data/sms.tsv`.

2. Iris Dataset (Task 3)

   Available in `sklearn.datasets`.

3. Titanic Dataset (Tasks 5–9)

   * Kaggle link: [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
   * Raw `train.csv` must be downloaded to the working directory.
   * Cleaned version (after imputation & encoding) provided as `train_.csv`.


🚀 Execution Steps

 Environment Setup

1. Clone the repository.
2. Create and activate a virtual environment (optional but recommended):

   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   
3. Install dependencies:

   pip install -r requirements.txt
 
 Running Notebooks

1. Launch Jupyter Notebook or JupyterLab in the project root:

   
   jupyter notebook
  
2. Open and execute each Task notebook in order: Task1 → Task2 → … → Task9.

Scripts (Optional)

Each notebook can be exported as a standalone script (`.py`) and run via:


python notebooks/TaskX_Name.py

🔖 Task Summaries

Task 1: Theory questions on Naive Bayes classifiers.
Task 2: SMS Spam detection with `MultinomialNB` and TF-IDF features.
Task 3: Classification on Iris dataset using `GaussianNB`, `DecisionTree`, `LogisticRegression`.
Task 4: Conceptual questions on Decision Trees (entropy, Gini, overfitting).
Task 5: Decision Tree on cleaned Titanic dataset (`train_cleaned.csv`), includes visualization.
Task 6: Hyperparameter tuning (`max_depth`, `min_samples_split`) with accuracy comparison plots.
Task 7: Conceptual questions on ensemble methods (Bagging vs Boosting, Random Forest variance, boosting weakness).
Task 8: Compare standalone Decision Tree vs `RandomForestClassifier`; plot feature importances.
Task 9: Train `AdaBoostClassifier` on Titanic, compare accuracy, F1, and training time against baselines.


📄 License

This assignment code is for educational purposes. No license.
