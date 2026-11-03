"""Compare logistic regression, Naive Bayes and SGD."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

Path('Evaluation/results').mkdir(exist_ok=True)

train = pd.read_csv('data/train.csv')
labels = train['true_category']
folds = StratifiedKFold(3, shuffle=True, random_state=42)

# Set up the three models.
models = [
    ('Logistic regression', LogisticRegression(max_iter=2000, random_state=42),
     {'classifier__C': [0.1, 1, 10], 'classifier__class_weight': [None, 'balanced']}),
    ('Naive Bayes', MultinomialNB(),
     {'classifier__alpha': [0.1, 1, 10], 'classifier__fit_prior': [True, False]}),
    ('SGD', SGDClassifier(loss='log_loss', max_iter=2000, random_state=42),
     {'classifier__alpha': [0.0001, 0.001, 0.01], 'classifier__class_weight': [None, 'balanced']}),
]
results = []

# Tune and test each model.
for name, classifier, settings in models:
    pipeline = Pipeline([
        ('words', TfidfVectorizer(stop_words='english')),
        ('classifier', classifier),
    ])
    settings['words__ngram_range'] = [(1, 1), (1, 2)]
    search = GridSearchCV(pipeline, settings, scoring='f1_macro', cv=folds)
    predictions = cross_val_predict(search, train['text'], labels, cv=folds)
    categories = sorted(labels.unique())
    matrix = pd.crosstab(labels, predictions).reindex(
        index=categories, columns=categories, fill_value=0,
    )
    filename = name.lower().replace(' ', '_')
    matrix.to_csv(f'Evaluation/results/classifier_{filename}_matrix.csv')
    results.append({
        'model': name,
        'correct': sum(predictions == labels),
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro'),
    })

pd.DataFrame(results).to_csv('Evaluation/results/classifier_comparison.csv', index=False)
