# === ASSIGNMENT 7: Text Analytics ===
# Operations: Tokenization, POS Tagging, Stop Words, Stemming, Lemmatization, TF-IDF

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# === Section 1: Download NLTK Resources ===
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# === Section 2: Setup ===
text = "Tokenization is the first step in text analytics. The process of breaking down text paragraphs into smaller chunks such as words or sentence is called tokenization."

stop_words = set(stopwords.words("english"))

# === Section 3: Sentence Tokenization ===
tokenized_text = sent_tokenize(text)
print("Sentence Tokens:", tokenized_text)
print("_______________________________________________________")

# === Section 4: Word Tokenization ===
tokenized_word = word_tokenize(text)
print("Word Tokens:", tokenized_word)
print("_______________________________________________________")

# === Section 5: Display Stop Words ===
print("Stop Words:", stop_words)
print("_______________________________________________________")

# === Section 6: Remove Non-Alphabetic Characters ===
text2 = "How to remove the non alphabetic 4charter 1244"
text2 = re.sub('[^a-zA-Z]', ' ', text2)
print("Cleaned Text:", text2)
print("_______________________________________________________")

# === Section 7: Stemming ===
e_words = ["wait", "waiting", "waited", "waits", "Learning"]
ps = PorterStemmer()
for w in e_words:
    print(f"Stem of '{w}' is '{ps.stem(w)}'")
print("_______________________________________________________")

# === Section 8: Lemmatization ===
wordnet_lemmatizer = WordNetLemmatizer()
text3 = "Studies studying cries cry"
for w in word_tokenize(text3):
    print("Lemma for '{}' is '{}'".format(w, wordnet_lemmatizer.lemmatize(w)))
print("_______________________________________________________")

# === Section 9: POS Tagging ===
data = "The pink sweater fit her perfectly"
words = word_tokenize(data)
for word in words:
    print(nltk.pos_tag([word]))
print("_______________________________________________________")

# === Section 10: Remove Stop Words ===
tokens = word_tokenize(text.lower())
filtered_words = [t for t in tokens if t not in stop_words]
print("Tokens          :", tokens)
print("Filtered Tokens :", filtered_words)
print("_______________________________________________________")

# === Section 11: TF-IDF Representation ===
documents = [
    "Text analytics is useful",
    "Text mining and text analytics",
    "Natural language processing uses text analytics"
]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(documents)

df_tfidf = pd.DataFrame(
    matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("TF-IDF Matrix:\n", df_tfidf)
print("_______________________________________________________")
