---
name: Advanced Regression Analysis
type: regression
version: 1.0.0
author: System
created: 2026-02-17
tags: [regression, prediction, continuous]
---

# Advanced Regression Analysis Skill

## Description
This skill enables the agent to perform comprehensive regression analysis on tabular data. It handles data preprocessing, multiple model training, hyperparameter tuning, and generates insightful visualizations.

## Capabilities
- ✅ Automatic data preprocessing and cleaning
- ✅ Missing value handling with smart strategies
- ✅ Feature encoding for categorical variables
- ✅ Multiple regression algorithms (Linear, Ridge, Lasso, Random Forest, XGBoost)
- ✅ Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- ✅ Ensemble methods (Voting, Stacking, Bagging)
- ✅ Model comparison and selection
- ✅ Comprehensive evaluation metrics (R², RMSE, MAE)
- ✅ Interactive visualizations
- ✅ Model persistence and saving

## Instructions
When this skill is activated, you are a regression analysis expert. Follow these steps:

1. **Data Understanding**
   - Examine data shape and structure
   - Identify numeric and categorical columns
   - Check for missing values and outliers
   - Understand feature distributions

2. **Data Preprocessing**
   - Handle missing values (mean/median for numeric, mode for categorical)
   - Encode categorical variables using LabelEncoder
   - Scale features using StandardScaler
   - Create interaction terms for correlated features

3. **Model Selection**
   - Train multiple regression models:
     * Linear Regression (baseline)
     * Ridge/Lasso (regularized)
     * Random Forest (ensemble)
     * Gradient Boosting (boosting)
     * SVR (support vector)
   - Use cross-validation (5-fold) for robust evaluation

4. **Hyperparameter Tuning**
   - For Random Forest: tune n_estimators, max_depth, min_samples_split
   - For Gradient Boosting: tune learning_rate, n_estimators, max_depth
   - Use GridSearchCV for small spaces, RandomizedSearchCV for large

5. **Ensemble Methods**
   - Create VotingRegressor from top 3 models
   - Try StackingRegressor with LinearRegression as final estimator
   - Compare ensemble performance with individual models

6. **Evaluation**
   - Calculate R², RMSE, MAE on test set
   - Generate actual vs predicted plots
   - Analyze residuals distribution
   - Check feature importance
   - Compare all models and select best

7. **Visualization**
   - Actual vs Predicted scatter plot
   - Residuals plot
   - Error distribution histogram
   - Feature importance bar chart
   - Model comparison bar chart

8. **Insights Generation**
   - Explain which features are most important
   - Discuss model performance and limitations
   - Provide recommendations for improvement
   - Suggest potential business applications

## Example Usage
User: "Predict house prices using this dataset with area, bedrooms, and location"
Assistant: [Activates regression skill and follows instructions]

## Input Requirements
- CSV file with numeric target column
- At least one feature column (numeric or categorical)
- Minimum 10 rows of data

## Output
- Trained model saved in models/ directory
- Comprehensive evaluation metrics
- Multiple visualizations
- Model comparison results
- Best model selection

## Compatible With
- Claude API
- Custom Agents
- Data Science Agent v2.0

## Version History
- 1.0.0 (2026-02-17): Initial release