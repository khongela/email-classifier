"""Check whether giving the subject more weight helps."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

Path('Evaluation/results').mkdir(exist_ok=True)

train = pd.read_csv('data/train.csv')
labels = train['true_category']

# Give the subject extra weight.
features = ColumnTransformer([
    ('combined', TfidfVectorizer(stop_words='english'), 'text'),
    ('subject', TfidfVectorizer(stop_words='english'), 'subject'),
], transformer_weights={'combined': 1, 'subject': 0})
pipeline = Pipeline([
    ('features', features),
    ('classifier', LogisticRegression(max_iter=2000, random_state=42)),
])
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
inner = StratifiedKFold(3, shuffle=True, random_state=42)
results = []

# Test each version on the same emails.
for name, weights in [('Default', []), ('Tuned', [0]), ('Extra subject weighting', [0.5, 1, 2])]:
    estimator = pipeline
    if weights:
        settings = {
            'features__combined__ngram_range': [(1, 1), (1, 2)],
            'features__transformer_weights': [{'combined': 1, 'subject': weight} for weight in weights],
            'classifier__C': [0.1, 1, 10],
            'classifier__class_weight': [None, 'balanced'],
        }
        estimator = GridSearchCV(pipeline, settings, scoring='f1_macro', cv=inner)

    held_out = []
    for fold, (training, validation) in enumerate(outer.split(train, labels)):
        estimator.fit(train.iloc[training], labels.iloc[training])
        held_out.append(pd.DataFrame({
            'repeat': fold // 5,
            'actual': labels.iloc[validation],
            'predicted': estimator.predict(train.iloc[validation]),
        }))

    predictions = pd.concat(held_out)
    categories = sorted(labels.unique())
    matrix = pd.crosstab(predictions.actual, predictions.predicted).reindex(
        index=categories, columns=categories, fill_value=0,
    )
    filename = name.lower().replace(' ', '_')
    matrix.to_csv(f'Evaluation/results/settings_{filename}_matrix.csv')
    for repeat, rows in predictions.groupby('repeat'):
        results.append({
            'model': name,
            'repeat': repeat + 1,
            'accuracy': accuracy_score(rows.actual, rows.predicted),
            'macro_f1': f1_score(rows.actual, rows.predicted, average='macro'),
        })


pd.DataFrame(results).groupby('model', sort=False)[['accuracy', 'macro_f1']].mean().to_csv(
    'Evaluation/results/settings_comparison.csv')
