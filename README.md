# IBM HR Analytics - Employee Attrition & Performance

## Overview
This project aims to analyze employee attrition and performance using the IBM HR Analytics dataset. The dataset provides information about employees in a company, including various features such as age, job role, department, performance ratings, etc. The goal is to build predictive models to understand factors contributing to employee attrition and predict attrition risk.

## Dataset
The dataset used in this project is obtained from IBM HR Analytics. It contains 1470 records and 35 features, including both numerical and categorical variables. Some of the key features include:

- Age
- Gender
- Marital Status
- Job Role
- Department
- Work-Life Balance
- Job Satisfaction
- Attrition (target variable)

## Analysis Steps
1. **Data Preprocessing**: 
    - Dropping irrelevant columns such as EmployeeNumber, EmployeeCount, Over18, StandardHours.
    - Encoding categorical variables using LabelEncoder or One-Hot Encoding.
    - Combining numerical and encoded categorical variables.
    - Splitting data into train and test sets.

2. **Model Building**:
    - Training multiple models including AdaBoost, Decision Tree, Random Forest, and Gradient Boosting.
    - Tuning hyperparameters using GridSearchCV to improve model performance.
    - Evaluating models using accuracy score, confusion matrix, and classification report.

3. **Model Comparison**:
    - Comparing the accuracies of different models using bar plots.
    - Highlighting the best-performing model based on accuracy.

4. **Feature Importance**:
    - Analyzing feature importance using algorithms like AdaBoost and Gradient Boosting.
    - Visualizing feature importance using bar plots.

## Results
- The project successfully built predictive models to analyze employee attrition and performance.
- The AdaBoost model achieved the highest accuracy of 87%, outperforming other models.
- Key factors contributing to employee attrition were identified, including age, job role, and work-life balance.

## Future Work
- Explore additional machine learning algorithms and ensemble methods for improved prediction.
- Gather more data or additional features to enhance model performance.
- Implement the models into production for real-time attrition prediction and risk management.

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Jupyter Notebook `Employee_Attrition_Analysis.ipynb` to reproduce the analysis.
4. Explore the code and modify it as needed for your specific use case.

## Credits
- IBM HR Analytics for providing the dataset.
- Contributors: Atharva Kate
