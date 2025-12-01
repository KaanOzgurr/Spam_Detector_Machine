import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


HAM_DIR = "data/ham"
SPAM_DIR = "data/spam"

def load_emails_from_folder(folder, label):
    emails = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            try:
                with open(path, encoding='latin-1') as f:
                    text = f.read().strip()
                if text:
                    emails.append((text, label))
            except Exception:
                continue
    return emails


ham_emails = load_emails_from_folder(HAM_DIR, 0)   # 0 = ham
spam_emails = load_emails_from_folder(SPAM_DIR, 1) # 1 = spam

print("Ham emails loaded:", len(ham_emails))
print("Spam emails loaded:", len(spam_emails))

all_emails = ham_emails + spam_emails
df = pd.DataFrame(all_emails, columns=['text', 'label'])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean'] = df['text'].apply(clean_text)


vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_df=0.9)
X = vectorizer.fit_transform(df['clean'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


while True:
    email = input("\nEnter full email content (or 'q' to quit):\n")
    if email.lower() == 'q':
        print("Exiting.")
        break
    email_clean = clean_text(email)
    X_new = vectorizer.transform([email_clean])
    pred = model.predict(X_new)[0]
    if pred == 1:
        print("⚠️ This email is classified as SPAM.")
    else:
        print("✅ This email is classified as HAM (legitimate).")






