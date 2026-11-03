from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

dataset = pd.read_csv('data/train.csv', dtype={'email_id': str})
x = dataset['text']
y = dataset['true_category']

Path('outputs').mkdir(exist_ok=True)

# Turn the text into numbers and train the classifier.
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', LogisticRegression(max_iter=2000, random_state=42)),
])

# Set the options to try.
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__class_weight': [None, 'balanced'],
}

inner_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = GridSearchCV(
    model,
    parameters,
    cv=inner_folds,
    scoring='f1_macro',
    n_jobs=1,
    error_score='raise',
)

# Choose the best settings and train on all labelled emails.
search.fit(x, y)

model = search.best_estimator_

joblib.dump(model, 'outputs/email_classifier.joblib')
print('Selected settings:', search.best_params_)
print('Saved tuned logistic regression without calibration.')
