import joblib
import pandas as pd

# Load the model and predict the email categories.
test = pd.read_csv('data/test.csv', dtype={'email_id': str})
model = joblib.load('outputs/email_classifier.joblib')
probabilities = model.predict_proba(test['text'])

rows = []

# Pick the category with the highest score.
for email_index in range(len(test)):
    email = test.iloc[email_index]
    email_probabilities = probabilities[email_index]

    best_index = email_probabilities.argmax()
    best_category = model.classes_[best_index]
    best_probability = email_probabilities[best_index]

    result = {
        'email_id': email['email_id'],
        'predicted_category': best_category,
        'confidence_score': best_probability,
    }
    rows.append(result)

results = pd.DataFrame(rows)
results.to_csv('outputs/results.csv', index=False)
print(results.to_string(index=False))
print('Saved outputs/results.csv')
