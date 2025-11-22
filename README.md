# 📊 Diabetes Prediction Using KNN

This project is dedicated to predicting diabetes using the **K-Nearest Neighbors (KNN)** algorithm. The model determines whether a person has diabetes based on clinical features from the **diabetes.csv** dataset. The project covers data loading, analysis, visualization, model building, and selecting the optimal `k` value.

---

## 📁 Dataset Information

The dataset consists of the following columns:

| Column Name | Description |
|-----------|---------|
| **Pregnancies** | Number of pregnancies |
| **Glucose** | Blood glucose level |
| **BloodPressure** | Blood pressure |
| **SkinThickness** | Skin thickness (mm) |
| **Insulin** | Insulin level |
| **BMI** | Body Mass Index |
| **DiabetesPedigreeFunction** | Genetic diabetes likelihood |
| **Age** | Age |
| **Outcome** | Diabetes presence (1 – yes, 0 – no) |

---

## 🔎 Project Structure

### **1. Data Loading**
The `diabetes.csv` file is read using `pandas` and initial data analysis is performed.

### **2. Data Visualization**
The following visualizations are created:

- **Countplot** of Outcome
- **Correlation heatmap** of features
- **Glucose vs Outcome** boxplot
- **BMI vs Outcome** boxplot
- **Pairplot** of key features

### **3. Standardization**
All features are scaled using `StandardScaler`.  
This is essential for the KNN algorithm to perform correctly.

### **4. Train/Test Split**
The dataset is split as follows:

- **80% — Train**
- **20% — Test**

### **5. Building the KNN Model**

The model is defined as:

```python
KNeighborsClassifier(n_neighbors=11)
```

---

### **6. Model Evaluation**

The model is evaluated using the following metrics:

- **Jaccard Score**
- **Confusion Matrix**
- **Classification Report**

These metrics allow a detailed analysis of the model's accuracy and performance.

---

### **7. Finding the Best k**

`GridSearchCV` is used to test `k` values from 1 to 25:

- **best_params_** — best k value  
- **best_score_** — highest accuracy  
- **k ranking graph** — shows performance for different k values

---

### **8. Test Results Visualization**

Test results based on **standardized Glucose and BMI**:

- Displayed as a **scatter plot**  
- Helps to visually assess how well the model separates classes

---

## 🧪 Libraries Used

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  

---

## ▶️ How to Run the Project

1. Place the `diabetes.csv` file in the project folder.  
2. Run the code using Python or Jupyter Notebook.  
3. All visualizations, model results, and the optimal **k** value will be automatically generated.

---
