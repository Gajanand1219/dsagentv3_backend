---
name: Advanced Classification Analysis
type: classification
version: 1.0.0
author: System
created: 2026-02-17
tags: [classification, categorical, prediction]
---

# Advanced Classification Analysis Skill

## Description
This skill enables the agent to perform comprehensive classification analysis on tabular data. It handles class imbalance, multiple algorithms, and generates confusion matrices and ROC curves.

## Capabilities
- ✅ Automatic target encoding
- ✅ Class imbalance detection
- ✅ Multiple classifiers (Logistic Regression, Random Forest, SVM, KNN, XGBoost)
- ✅ Stratified train-test split
- ✅ Hyperparameter tuning
- ✅ Ensemble voting classifier
- ✅ Confusion matrix visualization
- ✅ ROC curves for binary classification
- ✅ Precision, Recall, F1 metrics

## Instructions
When this skill is activated, follow these steps:
1. Check class distribution and balance
2. Encode target variable if needed
3. Use stratified split to preserve class ratios
4. Train multiple classifiers with cross-validation
5. Tune hyperparameters for best models
6. Create voting ensemble from top performers
7. Generate confusion matrix and classification report
8. Plot ROC curves for binary cases
9. Select best model based on F1 score

## Compatible With
- Claude API
- Custom Agents