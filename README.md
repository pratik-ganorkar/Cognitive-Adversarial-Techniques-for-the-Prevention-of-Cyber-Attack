# Cognitive Adversarial Technique For Prevention Of Cyber Attack


With the rise in data-driven organisations worldwide, large amounts of information are inevitably transferred,relocated and consigned on networks. These operations are done on the Internet and within organisational intranets and may often contain information of high sensitivity. Thus, it becomes inevitable for companies and institutes to deploy network security protocols and strategies to counter 
cyber criminals and maintain the integrity and confidentiality of information and the availability of infrastructure. This repository presents a structured, efficient and coherent approach to 
**detect and prevent cyber attacks** and breaches within a complex organisational network environment. It discusses the idea of **ensemble learning** (a subdomain of Machine Learning) to create a stacking classifier that will **predict the type of network attack the network infrastructure** is most susceptible to. We achieve this by analysing numerous network parameters. The model classifies the network probability of being susceptible to Denial of Service (DOS), Impersonation, and Man-in-the-Middle (MITM) attacks. It is classified as Normal if none of the aforementioned attacks are detected as a threat.
The model is trained using the OPCUA dataset and has proved to be a promising tool for network defensive evaluation.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Training and Testing the Model](#training-and-testing-the-model)
5. [Evaluation and Results](#evaluation-and-results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Overview

Network infrastructures nowadays have become more comprehensive, extensive and composite than ever before. This large-scale and complex deployment makes them more defenceless against sophisticated cyber 
attacks often carried out by Black Hat Hackers, Hacktivists or even nation-states. The crucial apprehensions discussed in this are Denial of Service(DOS) attacks, Spoofing/Impersonation attacks and 
man in-the-middle (MITM) attacks. These attacks are used for disrupting services and for the pilferage and manipulation of critical and valuable information. Traditionally, these threats were detected using conventional approaches of rule-based filtering. However, as the threats began to evolve with sophistication, the traditional methods generally proved to be as inefficient. Hence, this suggests the need for advanced approaches like Machine Learning. **The aim is to propose the Machine Learning approach based on ensemble learning**. It involves traininga stacking classifier to anticipate the kinds of attacks that the network is most vulnerable to. In Ensemble Learning, a model type known as a stacking classifier generates predictions by combining multiple algorithms. The proposed approach uses Decision Trees, Support 
Vector Classifier and K-Nearest Neighbour as base models and Logistic Regression as the meta-model.

---

## Prerequisites

Before setting up the project, make sure you have the following dependencies installed on your machine:
- **Python**
- **pip** (Python package installer)
- Refer to below link
  [Requirements]([https://github.com/pratik-ganorkar/Cognitive-Adversarial-Technique-For-Prevention-Of-Cyber-Attack/blob/main/requirements.txt])

---

## Environment Setup

To set up environment in **local sysytem** install all the dependencies in your system and also give path if required.

To set up the environment for **Google Colab** Create an account and install/import the necessary dependencies.

 **Check the Repository** to understand the necessary python packages and libraries reqired for training and testing
 
```
  https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack-1
```
---


## Training and Testing the Model
Once the environment is set up and the dependencies are installed, you can run the application.

### Saving and Running the Code.

1. Open **Colab** on your system.
2. Copy the following code and paste it:

```
#Import

import pandas as pd
import numpy as np

df = pd.read_csv('/content/OPCUA_dataset_public.csv')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X = df.drop('multi_label', axis=1)
X = df.drop('service', axis=1)

y = df['multi_label']
y = df['service']

X = X.select_dtypes(include=['number'])
X = X.fillna(X.mean())  # Fill missing values with the mean

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models
base_models = [
    ('decision_tree', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True))  # SVC needs probability=True for stacking
]

# Define meta-model
meta_model = LogisticRegression()

# Create the Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Cross-validation folds for base models
)

# Train the stacking classifier
stacking_clf.fit(X_train, y_train)

# Make predictions
y_pred = stacking_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Stacking Classifier: {accuracy:.2f}')

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold, cross_val_score


# Create the stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Standardize the data
    ('stacking', stacking_clf)     # Apply the stacking classifier
])

# Define k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the stacking classifier using cross-validation
kfold_cv_scores = cross_val_score(stacking_clf, X, y, cv=kf)

# Print the results
print("Cross-validation scores:", kfold_cv_scores)
print("Mean cross-validation score:", kfold_cv_scores.mean())

from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score

# Define custom scoring functions
f1_scorer = make_scorer(f1_score, average='macro')
recall_scorer = make_scorer(recall_score, average='macro')
precision_scorer = make_scorer(precision_score, average='macro')
# Calculate F1-score using cross-validation
f1_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=f1_scorer)

# Calculate Sensitivity (Recall) using cross-validation
recall_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=recall_scorer)

# Calculate Precision using cross-validation
precision_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=precision_scorer)

# Print the results
print("F1-scores:", f1_scores)
print("Mean F1-score:", np.mean(f1_scores))
print("Sensitivity (Recall) scores:", recall_scores)
print("Mean Sensitivity (Recall):", np.mean(recall_scores))
print("Precision scores:", precision_scores)
print("Mean Precision:", np.mean(precision_scores))

# Create subplots for separate graphs
plt.figure(figsize=(15, 10))

# Plot F1-scores
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='o', color='blue')
plt.title('F1-score across Cross-Validation Folds')
plt.xlabel('Fold')
plt.ylabel('F1-score')
plt.grid(True)

# Plot Sensitivity (Recall) scores
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
plt.plot(range(1, len(recall_scores) + 1), recall_scores, marker='o', color='green')
plt.title('Sensitivity (Recall) across Cross-Validation Folds')
plt.xlabel('Fold')
plt.ylabel('Sensitivity (Recall)')
plt.grid(True)

# Plot Precision scores
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
plt.plot(range(1, len(precision_scores) + 1), precision_scores, marker='o', color='red')
plt.title('Precision across Cross-Validation Folds')
plt.xlabel('Fold')
plt.ylabel('Precision')
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

def evaluate_classifier(clf, X, y, skf, metric_name):
    f1_scorer = make_scorer(f1_score, average='macro')
    recall_scorer = make_scorer(recall_score, average='macro')
    precision_scorer = make_scorer(precision_score, average='macro')

    f1_scores = cross_val_score(clf, X, y, cv=skf, scoring=f1_scorer)
    recall_scores = cross_val_score(clf, X, y, cv=skf, scoring=recall_scorer)
    precision_scores = cross_val_score(clf, X, y, cv=skf, scoring=precision_scorer)

    print(f"{metric_name} F1-scores: {f1_scores}, Mean F1-score: {np.mean(f1_scores)}")
    print(f"{metric_name} Recall scores: {recall_scores}, Mean Recall: {np.mean(recall_scores)}")
    print(f"{metric_name} Precision scores: {precision_scores}, Mean Precision: {np.mean(precision_scores)}")

    return f1_scores, recall_scores, precision_scores
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# For each base classifier
for name, clf in base_models:
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), (name, clf)])
    f1_scores, recall_scores, precision_scores = evaluate_classifier(pipeline, X, y, skf, name)
    results[name] = (f1_scores, recall_scores, precision_scores)

# Evaluate meta-classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)
stacking_pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('stacking', stacking_clf)])
f1_scores, recall_scores, precision_scores = evaluate_classifier(stacking_pipeline, X, y, skf, 'meta_classifier')
results['meta_classifier'] = (f1_scores, recall_scores, precision_scores)
# Plot the results for each classifier
plt.figure(figsize=(15, 15))
colors = ['blue', 'green', 'red']

# Plot each base classifier and the meta-classifier
for idx, (name, (f1_scores, recall_scores, precision_scores)) in enumerate(results.items()):
    # F1-score plot
    plt.subplot(len(results), 3, idx * 3 + 1)
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='o', color=colors[0])
    plt.title(f'{name} - F1-score')
    plt.xlabel('Fold')
    plt.ylabel('F1-score')
    plt.grid(True)

    # Recall (Sensitivity) plot
    plt.subplot(len(results), 3, idx * 3 + 2)
    plt.plot(range(1, len(recall_scores) + 1), recall_scores, marker='o', color=colors[1])
    plt.title(f'{name} - Recall (Sensitivity)')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.grid(True)

    # Precision plot
    plt.subplot(len(results), 3, idx * 3 + 3)
    plt.plot(range(1, len(precision_scores) + 1), precision_scores, marker='o', color=colors[2])
    plt.title(f'{name} - Precision')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.grid(True)


plt.tight_layout()
plt.show()
```

## Evaluation and Results

Below are some example images of models working and results in the form of graph.

Base Leaner-Decision Tree

![Base Leaner-Decision Tree](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/result_dtree.png)) 

Base Leaner-KNN

![Base Leaner-KNN](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/result_knn.png))

Base Leaner-SVC

![Base Leaner-SVC](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/result_svc.png))

Base Learners Overall Performance

![Base Learners Overall Performance](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/result.png))

Base Learners Analysis

![Base Learners Analysis](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/base_learner_table.png))

Meta-Learners Overall Performance

![Search Functionality](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/result_meta.png))

Meta Learner Analysis

![Meta Learner Analysis]](https://github.com/pratik-ganorkar/Cognitive-Adversarial-Techniques-for-the-Prevention-of-Cyber-Attack/blob/main/meta_learner_table.png))

---

## Contributing

### Contributions are welcome! 

To contribute:
- Fork this repository.
- Create a feature branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -m 'Add your feature').
- Push to the branch (git push origin feature/your-feature).
- Open a pull request.

---

## License

---
   

