# 📊 Churn Prediction Analysis
## ✨ Introduction

This project focuses on analyzing customer churn in an e-commerce setting. The primary goal is to understand the factors contributing to customer churn and build a predictive model to identify customers at risk. By understanding these dynamics, targeted strategies can be developed to improve customer retention.

## 📁 Data Source

The analysis is based on the `churn_prediction.xlsx` dataset, which contains various customer attributes and their churn status.

## 🎯 Problem Statement

Customer churn poses a significant challenge for an e-commerce company, resulting in potential revenue loss. This project aims to build a predictive model to identify users at risk of churning, enabling the company to implement targeted promotion programs and retention strategies. Specifically, it seeks to answer:

1.  What are the main characteristics and behaviors of churned customers?
2.  Can we build a machine learning model to accurately predict customer churn?
3.  What actionable insights and recommendations can be derived to reduce churn?

## 📈 Goals & Objectives

*   **Data Quality Assessment**: Identify and handle missing values, duplicates, and outliers.
*   **Exploratory Data Analysis (EDA)**: Uncover key patterns and relationships between customer attributes and churn.
*   **Churn Driver Identification**: Pinpoint specific features that significantly influence churn likelihood.
*   **Predictive Model Development**: Train and evaluate machine learning models to forecast churn.
*   **Churned User Segmentation**: Group churned users to understand different types of churn behavior.
*   **Actionable Recommendations**: Provide data-driven strategies to mitigate churn.

## ⚙️ Methodology

### 1. Data Loading and Initial Inspection

The dataset `churn_prediction.xlsx` was loaded using pandas. Initial checks included `df.info()` for data types and non-null counts, and `df.head()` to preview the data.

### 2. Data Quality & Preprocessing

*   **Missing Values**: Identified missing values in `daysincelastorder`, `orderamounthikefromlastyear`, `tenure`, `ordercount`, `couponused`, `hourspendonapp`, and `warehousetohome`. These were handled during model preprocessing using imputation strategies (median for numerical, most frequent for categorical).
*   **Duplicates**: No duplicate rows were found.
*   **Class Imbalance**: The target variable `churn` showed imbalance, with approximately 83% non-churned (0) and 17% churned (1) customers. This was addressed in model training using `class_weight='balanced_subsample'`.
*   **Outliers**: Outliers were identified in numerical columns like `ordercount`, `couponused`, and `cashbackamount` using IQR method and visualized with boxplots.
*   **Categorical Feature Cleaning**: Consolidated similar categories (e.g., 'Mobile Phone' to 'Phone', 'CC' to 'Credit Card', 'COD' to 'Cash on Delivery') to ensure data consistency.

### 3. Exploratory Data Analysis (EDA)

Focused analysis on features identified as important by an initial Random Forest model:

*   **Tenure**: Churn is heavily concentrated among new users (0-2 months), suggesting an activation problem rather than a long-term loyalty issue. Churn rate drops significantly after the initial period.
*   **Cashback Amount**: Lower cashback amounts correlate with higher churn rates, indicating cashback as a potential retention lever.
*   **Complain**: Customers who complained showed a significantly higher churn rate (~31.7%) compared to those who didn't (~10.9%), highlighting the impact of service experience.
*   **Days Since Last Order**: Highest churn rates were observed for customers with very recent last orders (<=1 day), reinforcing the new-user churn problem. For longer-tenure users, inactivity alone was not a strong churn signal.
*   **Warehouse to Home Distance**: Churn rate increases with greater distance from the warehouse to the customer's home, suggesting logistics and delivery experience play a role. This effect was amplified for customers who also complained.
*   **Interaction between Tenure and Days Since Last Order**: A heatmap revealed that churn is extremely high for new users (tenure <= 2 months) across all levels of inactivity, but drops sharply for more seasoned customers.

### 4. Model Training and Evaluation

*   **Data Split**: The data was split into training (70%), validation (15%), and test (15%) sets, with stratification to maintain class distribution.
*   **Preprocessing Pipelines**: Separate `ColumnTransformer` pipelines were created for tree-based models (imputation only) and linear models (imputation + scaling).
*   **Baseline Model**: A `DummyClassifier` (most frequent strategy) was used as a baseline, yielding expectedly low scores (balanced_acc: 0.5, PR-AUC: 0.168).
*   **Model Comparison**: Several models were trained and evaluated on the validation set:
    *   Logistic Regression
    *   Decision Tree
    *   Random Forest
    *   Gradient Boosting

    Random Forest showed the best performance on the validation set, particularly in `balanced_accuracy` and `pr_auc`.

| model              | precision | recall    | balanced_accuracy | roc_auc   | pr_auc    |
| :----------------- | :-------- | :-------- | :---------------- | :-------- | :-------- |
| RandomForest       | 0.966667  | 0.816901  | 0.905606          | 0.992422  | 0.970075  |
| LogisticRegression | 0.469231  | 0.859155  | 0.831427          | 0.909172  | 0.728094  |
| DecisionTree       | 0.455598  | 0.830986  | 0.815208          | 0.881729  | 0.641258  |
| GradientBoosting   | 0.837838  | 0.654930  | 0.814663          | 0.947068  | 0.803314  |

### 5. Random Forest Model Enhancement

*   **Hyperparameter Tuning**: `GridSearchCV` was applied to the Random Forest model using `balanced_accuracy` as the scoring metric. The best parameters found were `{'model__bootstrap': False, 'model__max_depth': 20, 'model__min_samples_leaf': 2, 'model__min_samples_split': 2, 'model__n_estimators': 200}`.
*   **Final Evaluation**: The tuned Random Forest model was evaluated on the unseen test set.

    ```
    Tuned RF (TEST) PR-AUC: 0.9758
    TEST Confusion matrix: [[696   7]
                            [ 16 126]]
                  precision    recall  f1-score   support

               0     0.9775    0.9900    0.9837       703
               1     0.9474    0.8873    0.9164       142

        accuracy                         0.9728       845
       macro avg     0.9624    0.9387    0.9501       845
    weighted avg     0.9725    0.9728    0.9724       845
    ```
    The model achieved a PR-AUC of 0.976, correctly identifying 126 churn users with only 7 false alarms and missing 16 actual churners. This performance is highly suitable for targeted retention campaigns.

### 6. Clustering Churned Users

*   **Data Preparation**: Only churned customers (`churn == 1`) were selected. Categorical features were one-hot encoded, and all features were scaled using `MinMaxScaler`. Missing values were imputed using the median strategy.
*   **Dimensionality Reduction**: PCA was applied to reduce dimensionality, with 3 components explaining approximately 39% of the variance. This suggested the data is high-dimensional and PCA is mainly for visualization.
*   **K-Means Clustering**: The Elbow method was used to determine the optimal number of clusters (`k`). The WCSS curve did not show a clear elbow, indicating that the cluster structure is not strongly separated in the current feature space. Further analysis (e.g., using more domain-specific features) would be needed to derive meaningful clusters.

## 💡 Key Findings & Insights

1.  **Early Churn is Dominant**: The most significant churn occurs within the first 0-2 months of a customer's tenure. This points to an activation and onboarding issue.
2.  **Cashback Impact**: Higher cashback amounts are strongly associated with lower churn rates, suggesting that financial incentives are an effective retention tool, especially for new users.
3.  **Complaint Resolution is Crucial**: Customers who register complaints are much more likely to churn, emphasizing the importance of prompt and effective customer service.
4.  **Logistics Matter**: Customers living farther from warehouses show higher churn, particularly when combined with service complaints, indicating potential issues with delivery or order fulfillment for these segments.
5.  **Tenure Context**: 'Days Since Last Order' is not a reliable churn indicator on its own; its predictive power is highly dependent on the customer's tenure.

## 🚀 Recommendations

1.  **Optimize Onboarding Experience**: Implement targeted interventions and personalized communication during the first 30-60 days to improve early customer engagement and product adoption.
2.  **Prioritize Complaint Resolution**: Establish clear protocols for rapid and effective handling of customer complaints. Consider offering proactive compensation (e.g., apology credits, free shipping) after issues are resolved to rebuild trust.
3.  **Strategic Cashback Allocation**: Use cashback incentives more intelligently, focusing on new users or those showing early signs of dissatisfaction (e.g., low cashback amounts, recent complaints) to maximize retention impact while managing costs.
4.  **Enhance Logistics for Distant Customers**: Improve communication regarding delivery times, offer flexible shipping options, and ensure robust support for customers residing farther from fulfillment centers, particularly if they experience issues.
5.  **Develop Churn Risk Scores**: Combine insights from tenure, cashback, complaints, and logistics data to create a comprehensive churn risk score that informs proactive retention efforts.

## 📊 Model Performance (Tuned Random Forest)

The final Random Forest model, after hyperparameter tuning, achieved excellent performance on the test set:

*   **PR-AUC**: 0.976
*   **Balanced Accuracy**: 0.939
*   **Precision (Churn)**: 0.947
*   **Recall (Churn)**: 0.887

This model is highly effective at identifying churn-prone customers, making it a valuable asset for implementing targeted retention strategies.

## 🛠️ Tools & Technologies

*   **Python** (Programming Language)
*   **Pandas** (Data Manipulation and Analysis)
*   **NumPy** (Numerical Operations)
*   **Matplotlib** (Data Visualization)
*   **Seaborn** (Statistical Data Visualization)
*   **Scikit-learn** (Machine Learning Library - for models, preprocessing, and metrics)

## 🏃‍♂️ How to Run

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install dependencies** (if in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
    (Note: a `requirements.txt` would typically list `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`)
3.  **Open the Jupyter Notebook**: You can open the `churn_prediction_analysis.ipynb` (or similar name) notebook in Jupyter Lab or VS Code.
4.  **Run the cells**: Execute all cells in the notebook sequentially to reproduce the analysis and model training.

## ✅ Conclusion

This project successfully identified key drivers of customer churn and developed a robust predictive model. The insights gained highlight the importance of early customer experience, effective complaint resolution, and targeted incentives in improving retention. The trained Random Forest model provides a powerful tool for proactive churn management, enabling businesses to retain valuable customers and optimize their strategies.
