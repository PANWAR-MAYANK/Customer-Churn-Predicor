# **Customer Churn Prediction with Machine Learning 📊📉**



## **Project Overview 📝**

Customer churn is a significant concern for businesses, especially in sectors such as telecommunications, where retaining customers can be costly. This project applies machine learning models to predict the likelihood of customer churn based on historical data provided by a telecom company.

The dataset contains various features about customers, including demographic data, account information, service usage, and customer support interactions. By identifying which features contribute most to churn, the model helps businesses take strategic actions to reduce churn rates.

---

## **Key Highlights 🌟**

- **Data Preprocessing**: The raw data underwent comprehensive cleaning, including handling missing values, converting categorical variables to numeric, and feature engineering.
  
- **Class Imbalance Handling**: Addressed class imbalance using **SMOTEENN** (Synthetic Minority Oversampling Technique + Edited Nearest Neighbour), ensuring the model is robust even in imbalanced datasets.
  
- **Machine Learning Models**: Implemented multiple classification models, including:
  - **Decision Tree Classifier** 🌳
  - **Random Forest Classifier** 🌲
  - **XGBoost Classifier** 🚀

- **PCA (Principal Component Analysis)**: Explored dimensionality reduction techniques (PCA), though it didn’t significantly improve the model performance in this case.

- **Model Evaluation**: Models were evaluated using performance metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**. A detailed classification report was provided for each model to assess their effectiveness in predicting churn.

---

## **Project Workflow 🏃‍♂️**

### **1. Data Exploration and Preprocessing 🔍**
The first step was to explore and clean the data. Key preprocessing steps included:

- **Data Cleaning**: Handling missing values, transforming features, and ensuring data quality.
- **Feature Engineering**: New features were created to help the models better understand the data, such as grouping `tenure` into bins and encoding categorical variables using one-hot encoding.
  
**Key preprocessing actions**:
- Handling missing values in columns like `TotalCharges` by converting them into numeric values.
- Encoding categorical variables such as `Contract`, `PaymentMethod`, `InternetService`, etc.
- Binning `tenure` into groups for better insights.
  
### **2. Class Imbalance Handling ⚖️**
Customer churn is often an imbalanced classification problem. To address this:

- **SMOTEENN** (Synthetic Minority Oversampling Technique + Edited Nearest Neighbour) was used to balance the classes. SMOTE generates synthetic samples for the minority class, while ENN removes poorly classified samples.

### **3. Model Training 🏋️‍♂️**
The following models were trained and evaluated:

- **Decision Tree Classifier** 🌳: A simple and interpretable model to understand decision-making.
- **Random Forest Classifier** 🌲: An ensemble method that averages multiple decision trees to improve predictive accuracy.
- **XGBoost Classifier** 🚀: A highly efficient gradient boosting algorithm that performed well on structured data like this one.
  
### **4. Model Evaluation 📈**
Each model was evaluated using various performance metrics:
- **Accuracy**: The overall proportion of correctly classified instances.
- **Precision**: The ability to not label a negative sample as positive.
- **Recall**: The ability to find all positive samples.
- **F1-Score**: The balance between precision and recall.

**Key insights**:
- After SMOTEENN, the performance significantly improved across all models.
- **XGBoost** and **Random Forest** performed similarly, so either can be used for deployment.

### **5. PCA (Principal Component Analysis) 🔄**
PCA was applied to reduce dimensionality and improve model performance. However, in this case, PCA did not yield a significant improvement in accuracy or other metrics, indicating that the original feature set was already well-structured for prediction.

---

## **Results 📊**

### **Decision Tree Model 🌳**
**Without Upsampling**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.87   | 0.86     | 1043    |
| 1     | 0.61      | 0.57   | 0.59     | 364     |
| **Accuracy** |  |  | 0.79 | 1407 |
| **Macro avg** | 0.73 | 0.72 | 0.73 | 1407 |
| **Weighted avg** | 0.79 | 0.79 | 0.79 | 1407 |

**With SMOTEENN Upsampling**:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.93   | 0.94     | 545     |
| 1     | 0.94      | 0.95   | 0.94     | 625     |
| **Accuracy** |  |  | 0.94 | 1170 |
| **Macro avg** | 0.94 | 0.94 | 0.94 | 1170 |
| **Weighted avg** | 0.94 | 0.94 | 0.94 | 1170 |

---

### **Random Forest Classifier 🌲**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.93   | 0.94     | 545     |
| 1     | 0.94      | 0.95   | 0.94     | 625     |
| **Accuracy** |  |  | 0.94 | 1170 |
| **Macro avg** | 0.94 | 0.94 | 0.94 | 1170 |
| **Weighted avg** | 0.94 | 0.94 | 0.94 | 1170 |

---

### **XGBoost Classifier 🚀**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.93   | 0.94     | 545     |
| 1     | 0.94      | 0.95   | 0.94     | 625     |
| **Accuracy** |  |  | 0.94 | 1170 |
| **Macro avg** | 0.94 | 0.94 | 0.94 | 1170 |
| **Weighted avg** | 0.94 | 0.94 | 0.94 | 1170 |

---

### **PCA (Principal Component Analysis) 🔄**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.94      | 0.93   | 0.94     | 545     |
| 1     | 0.94      | 0.95   | 0.94     | 625     |
| **Accuracy** |  |  | 0.94 | 1170 |
| **Macro avg** | 0.94 | 0.94 | 0.94 | 1170 |
| **Weighted avg** | 0.94 | 0.94 | 0.94 | 1170 |


---

## **Final Conclusion 🏁**

- **SMOTEENN** significantly improved the model performance by balancing the class distribution, which is crucial for imbalanced datasets like churn prediction.
- Among the models tested, **XGBoost** showed the best overall performance, followed closely by **Random Forest**. Either of these models can be deployed in real-world applications.
- **PCA** did not show significant improvements, but it’s worth considering for reducing computational complexity in larger datasets.

---

## **Future Work 🔮**

- **Hyperparameter Tuning**: Optimizing hyperparameters of the models (e.g., using GridSearchCV or RandomizedSearchCV) can further improve performance.
- **Deployment**: The model can be deployed as part of a real-time churn prediction system using a cloud platform.
- **Model Interpretability**: Techniques like **SHAP** values or **LIME** could be used to explain individual predictions, helping business users interpret the model’s decisions.
  
