# Sentiment Analysis of Website Reviews

This project focuses on performing sentiment analysis on a dataset of website reviews. The goal is to classify the reviews into positive or negative sentiments using various machine learning models such as Logistic Regression, Random Forest, Support Vector Classifier (SVC), and XGBoost. The models are evaluated and tuned using performance metrics such as accuracy, classification reports, and confusion matrices.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Installation and Setup](#installation-and-setup)
5. [Data Preprocessing](#data-preprocessing)
6. [Models](#models)
7. [Model Evaluation](#model-evaluation)
8. [Results](#results)
9. [Contributions](#contributions)
10. [License](#license)

## Project Overview

In this project, we perform sentiment analysis on a collection of website reviews. The dataset contains the website name, review text, and a sentiment label indicating whether the sentiment of the review is positive or negative. The text data is preprocessed to remove stopwords, punctuation, and apply stemming. Several machine learning models are used to predict sentiment, and hyperparameter tuning is performed to improve model performance.

### Key Steps in the Project:
1. **Data Loading**: We load training and testing datasets from CSV files.
2. **Data Preprocessing**: Text data is cleaned and transformed to remove unnecessary words and punctuation.
3. **Feature Representation**: We use techniques like Bag of Words (BoW) and TF-IDF to represent the text data numerically.
4. **Model Training**: Various classifiers such as Logistic Regression, Random Forest, SVC, and XGBoost are trained and evaluated.
5. **Model Evaluation**: Models are evaluated using accuracy, classification report, and confusion matrix.

## Technologies Used

- **Python** (for implementation)
- **pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **scikit-learn** (for machine learning models, feature extraction, and evaluation)
- **XGBoost** (for gradient boosting model)
- **nltk** (for natural language processing tasks)
- **matplotlib** (for data visualization)
- **seaborn** (for generating confusion matrix heatmaps)

## Dataset

The dataset used in this project contains website reviews with the following columns:

- **website_name**: Name of the website being reviewed.
- **text**: Review text.
- **is_positive_sentiment**: Sentiment label (0 for negative, 1 for positive).

The dataset is divided into training and testing sets: `x_train.csv`, `y_train.csv`, `x_test.csv`, and `y_test.csv`.

## Installation and Setup

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git

2. Install required dependencies
   ``` bash
   pip install -r requirements.txt
3. Ensure that the dataset files (``` x_train.csv, y_train.csv, x_test.csv, y_test.csv```) are placed in the ``` Dataset``` directory.

## Data Preprocessing

The following steps are applied during data preprocessing:

- Text data is converted to lowercase.
- Punctuation is removed.
- Stop words are filtered out.
- Stemming is applied using the PorterStemmer from NLTK.

## Models

The following models are used for sentiment classification:

- **Logistic Regression**: A simple linear model for binary classification.
- **Random Forest**: An ensemble model that combines multiple decision trees.
- **Support Vector Classifier (SVC)**: A model that finds the hyperplane that best separates the data into different classes.
- **XGBoost**: A gradient boosting algorithm for optimized performance.

### Hyperparameter Tuning

- **Logistic Regression**: Parameters tuned include regularization strength (`C`), penalty type (`l1` or `l2`), and solver type.
- **Random Forest**: Parameters tuned include number of estimators, maximum depth, and minimum samples for splitting and leaf nodes.
- **SVC**: Parameters tuned include regularization strength (`C`), kernel function, and kernel coefficient (`gamma`).
- **XGBoost**: Parameters tuned include the number of estimators, maximum depth of trees, learning rate, and subsampling ratio.

## Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Confusion Matrix**: A matrix showing the true positives, false positives, true negatives, and false negatives.

## Results

Each model is evaluated on the test set, and the evaluation results (accuracy, classification report, and confusion matrix) are printed for comparison.

## Contributions

Contributions are welcome! Feel free to fork the project, open issues, or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
