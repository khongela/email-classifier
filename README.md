# Email Classification for Financial Services

## About

This project classifies client emails into Account Management, Investment Advisory, Loan Processing, Insurance Claims or Other.

It uses TF-IDF and logistic regression to learn from 44 labelled emails and classify 12 unlabelled emails. The workflow reads the supplied HTML files, trains the model and saves each email’s predicted category and confidence score.

**Tools:** Python · pandas · Beautiful Soup · scikit-learn · Matplotlib

[Read the report](report.pdf)

## Setup and run

Install **Python 3.12** and **Git**, then open PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/khongela/email-classifier.git
cd email-classifier
```

### 2. Set up the environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run the classifier

```powershell
.\.venv\Scripts\python.exe run.py
```

Run all commands from the repository root, where `run.py` is located. The supplied emails and labels are in `data/`.

### Output

The workflow runs ingestion, modelling and reporting in order and saves the final predictions to **`outputs/results.csv`**.

```text
email_id,predicted_category,confidence_score
```

The trained model is saved to `outputs/email_classifier.joblib`. Confidence is the highest predicted class probability.

## Results

Logistic regression performed best in the initial comparison against Multinomial Naive Bayes and SGD. Tuning improved its performance, while giving the subject line extra weight did not.

| Final model | Mean accuracy | Mean macro F1 |
|---|---:|---:|
| Tuned TF-IDF + logistic regression | 92.73% | 0.898 |

[View the final predictions CSV](outputs/results.csv) for all 12 emails, including their IDs, predicted categories and confidence scores.

**Other remains the main weakness.** Only half of its messages were identified across the repeated evaluation. Staff should check routes during a pilot, with particular attention to messages outside the specialist categories.

These results come from a small dataset used during development. A larger independent test set is needed to confirm performance on new emails.

## Reproduce the evaluation

<details>
<summary>Show evaluation and chart commands</summary>

After running the classifier, compare the models:

```powershell
.\.venv\Scripts\python.exe Evaluation/compare_classifiers.py
.\.venv\Scripts\python.exe Evaluation/compare_subject_weighting.py
```

Install the plotting packages and generate the charts:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-figures.txt
.\.venv\Scripts\python.exe Evaluation/export_figures.py
```

- **Scores and matrices:** `Evaluation/results/`
- **Charts:** `Evaluation/figures/`

The classifier comparison evaluates each email once. The settings comparison evaluates each email five times.

</details>

## Next steps

- Collect more verified emails, especially examples of Other.
- Use staff corrections to improve the training data.
- Add context from email threads, attachments and sender details.
- Compare MiniLM embeddings with TF-IDF to test whether they improve classification.
- Explore calibration to test whether the probability scores better match observed correctness.

