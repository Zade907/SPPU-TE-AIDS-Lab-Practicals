import nltk
import math
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# download once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# sample document
text = """Natural Language Processing is a field of Artificial Intelligence 
that focuses on interaction between computers and humans. 
It helps machines understand, interpret and generate human language."""

# 1 Tokenization
tokens = word_tokenize(text)
print("Tokens:\n", tokens)

# 2 POS Tagging
pos = pos_tag(tokens)
print("\nPOS Tags:\n", pos)

# 3 Stopword removal
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]

print("\nAfter Stopword Removal:\n", filtered)

# 4 Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]

print("\nStemmed Words:\n", stemmed)

# 5 Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

print("\nLemmatized Words:\n", lemmatized)

# ---------- TF IDF ----------

documents = [
    "Natural language processing is fun",
    "Machine learning and language processing",
    "Artificial intelligence and machine learning"
]

# tokenize docs
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Term Frequency
tf = []
for doc in tokenized_docs:
    counter = Counter(doc)
    total = len(doc)
    tf.append({word: count/total for word, count in counter.items()})

print("\nTerm Frequency:")
for i in tf:
    print(i)

# Inverse Document Frequency
N = len(tokenized_docs)

idf = {}
all_words = set(word for doc in tokenized_docs for word in doc)

for word in all_words:
    containing = sum(1 for doc in tokenized_docs if word in doc)
    idf[word] = math.log(N / containing)

print("\nInverse Document Frequency:")
print(idf)