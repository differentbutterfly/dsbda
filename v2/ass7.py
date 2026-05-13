import nltk
import re
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

for pkg in [
    'punkt',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]:
    nltk.download(pkg)

text = """
Tokenization is the first step in text analytics.
The process of breaking down text paragraphs into smaller chunks
such as words or sentences is called tokenization.
Text analytics is very useful in Natural Language Processing.
"""

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

choice = 1

while choice != 10:
    print("""
------------- MENU -------------
1. Sentence Tokenization
2. Word Tokenization
3. Display Stop Words
4. Remove Non-Alphabetic Characters
5. Stemming
6. Lemmatization
7. POS Tagging
8. Remove Stop Words from Text
9. TF-IDF Representation
10. Exit
""")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        print("Sentence Tokenization:\n", sent_tokenize(text))

    elif choice == 2:
        print("Word Tokenization:\n", word_tokenize(text))

    elif choice == 3:
        print("Stop Words:\n", stop_words)

    elif choice == 4:
        print("Cleaned Text:\n", re.sub('[^a-zA-Z]', ' ', text))

    elif choice == 5:
        words = ["wait", "waiting", "waited", "waits", "learning"]

        for w in words:
            print(w, "->", ps.stem(w))

    elif choice == 6:
        text2 = "studies studying cries cry"

        for w in word_tokenize(text2):
            print(w, "->", lemmatizer.lemmatize(w))

    elif choice == 7:
        data = "The pink sweater fit her perfectly"
        print("POS Tagging:\n", nltk.pos_tag(word_tokenize(data)))

    elif choice == 8:
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w not in stop_words and w.isalpha()]

        print("Original Tokens:\n", tokens)
        print("Filtered Tokens:\n", filtered)

    elif choice == 9:
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

    elif choice == 10:
        print("Program Ended Successfully")

    else:
        print("Invalid Choice")