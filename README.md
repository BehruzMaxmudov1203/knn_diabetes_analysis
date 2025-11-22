# 📊 Diabetes Prediction Using KNN

This project predicts diabetes using the **K-Nearest Neighbors (KNN)** algorithm. The model determines whether a person has diabetes based on clinical features from the **diabetes.csv** dataset. The project covers data loading, analysis, visualization, model building, and selecting the optimal `k` value.

---

## 📁 Dataset Information

The dataset consists of the following columns:

| Column Name                  | Description                         |
| ---------------------------- | ----------------------------------- |
| **Pregnancies**              | Number of pregnancies               |
| **Glucose**                  | Blood glucose level                 |
| **BloodPressure**            | Blood pressure                      |
| **SkinThickness**            | Skin thickness (mm)                 |
| **Insulin**                  | Insulin level                       |
| **BMI**                      | Body Mass Index                     |
| **DiabetesPedigreeFunction** | Genetic diabetes likelihood         |
| **Age**                      | Age                                 |
| **Outcome**                  | Diabetes presence (1 – yes, 0 – no) |

Dataset link: [diabetes.csv](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 🔎 Project Structure

### **1. Data Loading**

The `diabetes.csv` file is read using [**pandas**](https://pandas.pydata.org/) and initial data analysis is performed.

### **2. Data Visualization**

Visualizations are created using [**matplotlib**](https://matplotlib.org/) and [**seaborn**](https://seaborn.pydata.org/):

* **Countplot** of Outcome
* **Correlation heatmap** of features
* **Glucose vs Outcome** boxplot
* **BMI vs Outcome** boxplot
* **Pairplot** of key features

### **3. Standardization**

All features are scaled using [**StandardScaler**](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) from scikit-learn. This is essential for the KNN algorithm to perform correctly.

### **4. Train/Test Split**

The dataset is split using [**train_test_split**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html):

* **80% — Train**
* **20% — Test**

### **5. Building the KNN Model**

The model is defined using [**KNeighborsClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html):

```python
KNeighborsClassifier(n_neighbors=11)
```

---

### **6. Model Evaluation**

The model is evaluated using metrics from [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html):

* **Jaccard Score** — [link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)
* **Confusion Matrix** — [link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
* **Classification Report** — [link](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

---

### **7. Finding the Best k**

`GridSearchCV` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) is used to test `k` values from 1 to 25:

* **best_params_** — best k value
* **best_score_** — highest accuracy
* **k ranking graph** — shows performance for different k values

---

### **8. Test Results Visualization**

Test results based on **standardized Glucose and BMI**:

* Displayed as a **scatter plot**
* Helps to visually assess how well the model separates classes

---

## 🧪 Libraries Used

* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)

---

## ▶️ How to Run the Project

1. Place the `diabetes.csv` file in the project folder.
2. Run the code using Python or Jupyter Notebook.
3. All visualizations, model results, and the optimal **k** value will be automatically generated.
