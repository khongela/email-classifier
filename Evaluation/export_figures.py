"""Create the confusion matrices and comparison charts for the report."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'svg.fonttype': 'none',
})
Path('Evaluation/figures').mkdir(exist_ok=True)


def save_figure(figure, name):
    figure.savefig(f'Evaluation/figures/{name}.svg', bbox_inches='tight')
    if name in ['classifier_comparison', 'settings_comparison', 'settings_tuned_matrix']:
        figure.savefig(f'Evaluation/figures/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(figure)


# Draw one confusion matrix for each model.
for path in sorted(Path('Evaluation/results').glob('*_matrix.csv')):
    if not path.name.startswith(('classifier_', 'settings_')):
        continue

    matrix = pd.read_csv(path, index_col=0)
    categories = [name.replace(' ', '\n') for name in matrix.columns]
    figure, axis = plt.subplots(figsize=(8, 7), layout='constrained')

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix.to_numpy(),
        display_labels=categories,
    )
    display.plot(ax=axis, cmap='Blues', values_format='d', colorbar=True)
    axis.set_title('Confusion Matrix', pad=20)
    axis.set_xlabel('Prediction', labelpad=10)
    axis.set_ylabel('Actual', labelpad=10)
    axis.xaxis.set_label_position('top')
    axis.xaxis.tick_top()

    threshold = matrix.to_numpy().max() / 2
    for label, count in zip(display.text_.flat, matrix.to_numpy().flat):
        label.set_color('white' if count > threshold else 'black')

    save_figure(figure, path.stem)


# Compare accuracy and macro F1 using the same scale.
for filename, title in [
    ('classifier_comparison', 'Classifier comparison'),
    ('settings_comparison', 'Logistic regression settings'),
]:
    scores = pd.read_csv(f'Evaluation/results/{filename}.csv')
    figure, axis = plt.subplots(figsize=(8, 4.5), layout='constrained')
    positions = list(range(len(scores)))
    width = 0.35

    accuracy_bars = axis.bar(
        [position - width / 2 for position in positions],
        scores['accuracy'], width, label='Accuracy', color='#1d5ca1',
    )
    f1_bars = axis.bar(
        [position + width / 2 for position in positions],
        scores['macro_f1'], width, label='Macro F1', color='#29964a',
    )

    axis.bar_label(accuracy_bars, fmt='%.3f', padding=4, color='black')
    axis.bar_label(f1_bars, fmt='%.3f', padding=4, color='black')
    axis.set_xticks(positions, scores['model'])
    axis.set_ylim(0, 1.15)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axis.set_title(title, pad=35)
    axis.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
    axis.spines[['top', 'right']].set_visible(False)

    save_figure(figure, filename)

print('Saved eight SVG charts in Evaluation/figures/.')
