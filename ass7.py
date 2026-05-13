import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

text = "Tokenization is the first step in text analytics. The process of breaking down text paragraphs into smaller chunks such as words or sentence is called tokenization."

stop_words = set(stopwords.words("english"))

choice = 1
while(choice != 10):
    print("1. Sentence Tokenization")
    print("2. Word Tokenization")
    print("3. Print Stop Words")
    print("4. Remove non alphabetic chars from sentence")
    print("5. Stemming")
    print("6. Lemmatization")
    print("7. POS Tagging")
    print("8. Remove stop words from text")
    print("9. TF-IDF")
    print("10. Exit")
    choice = int(input("Enter your choice: "))

    if choice == 1:
        tokenized_text = sent_tokenize(text)
        print("Sentence Tokens:", tokenized_text)
        print("_______________________________________________________")
    if choice == 2:
        tokenized_word = word_tokenize(text)
        print("Word Tokens:", tokenized_word)
        print("_______________________________________________________")
    if choice == 3:
        print("Stop Words:", stop_words)
        print("_______________________________________________________")
    if choice == 4:
        text2 = "How to remove the non alphabetic 4charter 1244"
        text2 = re.sub('[^a-zA-Z]', ' ', text2)
        print("Cleaned Text:", text2)
        print("_______________________________________________________")
    if choice == 5:
        e_words = ["wait", "waiting", "waited", "waits", "Learning"]
        ps = PorterStemmer()
        for w in e_words:
            print(f"Stem of '{w}' is '{ps.stem(w)}'")
        print("_______________________________________________________")
    if choice == 6:
        wordnet_lemmatizer = WordNetLemmatizer()
        text3 = "Studies Studying cries cry"
        for w in word_tokenize(text3):
            print("Lemma for '{}' is '{}'".format(w, wordnet_lemmatizer.lemmatize(w)))
        print("_______________________________________________________")
    if choice == 7:
        data = "The pink sweater fit her perfectly"
        words = word_tokenize(data)
        for word in words:
            print(nltk.pos_tag([word]))
        print("_______________________________________________________")
    if choice == 8:
        tokens = word_tokenize(text.lower())
        filtered_words = [t for t in tokens if t not in stop_words]
        print("Tokens          :", tokens)
        print("Filtered Tokens :", filtered_words)
        print("_______________________________________________________")
    if choice == 9:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pandas as pd

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
    if choice == 10:
        break
