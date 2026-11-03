from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

def read_email(path):
    """
    Read an email from an HTML file and remove scripts and styles.
    Combine the subject and body into one text field and return it
    with the filename, email ID and subject.
    """
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    subject = soup.select_one('[data-field="subject"]')
    body = soup.select_one('.email-body')
    email_id = soup.select_one('[data-field="email_id"]')
    
    if subject is None or body is None or email_id is None:
        raise ValueError(f'Missing fields in {path.name}')

    for item in body.select('script, style, title'):
        item.decompose()

    subject_text = subject.get_text(' ', strip=True)
    
    body_text = body.get_text(' ', strip=True)

    return {
        'filename': path.name,
        'subject': subject_text,
        'email_id': email_id.get_text(strip=True),
        'text': f'{subject_text} {body_text}',
    }

# Match each email to its label.
labels = pd.read_csv('data/train_labels.csv')
emails = pd.DataFrame([
    read_email(path) for path in sorted(Path('data/train').glob('*.html'))
])
test = pd.DataFrame([
    read_email(path) for path in sorted(Path('data/test').glob('*.html'))
])

train = labels.merge(emails, on='filename', validate='one_to_one')
if len(train) != 44 or len(test) != 12:
    raise ValueError('Expected 44 labelled emails and 12 unlabelled emails.')

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
print(f'Prepared {len(train)} labelled emails and {len(test)} unlabelled emails.')
