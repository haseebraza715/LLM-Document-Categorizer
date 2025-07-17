import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
import nltk

# Download NLTK stopwords if not already present
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Load dataset
print("Loading 20 Newsgroups...")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
labels = newsgroups.target
target_names = newsgroups.target_names

# Step 2: Basic cleaning function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)                  # remove newlines/tabs
    text = re.sub(r'[^\w\s]', '', text.lower())       # lowercase and remove punctuation
    text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
    return text.strip()

# Step 3: Clean all documents
print("Cleaning documents...")
cleaned_docs = [clean_text(doc) for doc in documents]

# Step 4: Save to CSV for reuse
df = pd.DataFrame({
    'text': cleaned_docs,
    'label_id': labels,
    'label_name': [target_names[i] for i in labels]
})
df.to_csv('20newsgroups_cleaned.csv', index=False)

print("Done! Cleaned dataset saved as 20newsgroups_cleaned.csv")
print(df.head()) 