# === ASSIGNMENT 11: Word Count Application ===
# Python equivalent of Hadoop MapReduce WordCount
# Counts word occurrences in a given text using Python collections

from collections import Counter
import re

# === Section 1: Sample Input Text ===
sample_text = """
Tokenization is the first step in text analytics.
The process of breaking down text paragraphs into smaller chunks
such as words or sentences is called tokenization.
Text analytics is very useful in Natural Language Processing.
Tokenization helps in understanding the structure of text.
"""

print("========== INPUT TEXT ==========")
print(sample_text)

# === Section 2: Tokenization (Splitting into words) ===
words = re.findall(r'\b\w+\b', sample_text.lower())
print("========== TOKENIZED WORDS ==========")
print(words)

# === Section 3: Word Count using Counter ===
word_counts = Counter(words)
print("\n========== WORD COUNT (MapReduce Style) ==========")
print(f"{'Word':<20} {'Count':<10}")
print("-" * 30)
for word, count in sorted(word_counts.items()):
    print(f"{word:<20} {count:<10}")

# === Section 4: Top 10 Most Frequent Words ===
print("\n========== TOP 10 MOST FREQUENT WORDS ==========")
print(f"{'Word':<20} {'Count':<10}")
print("-" * 30)
for word, count in word_counts.most_common(10):
    print(f"{word:<20} {count:<10}")

# === Section 5: Filter Stop Words ===
stop_words = {'the', 'is', 'in', 'of', 'a', 'an', 'and', 'to', 'for', 'this',
              'that', 'it', 'as', 'by', 'on', 'at', 'are', 'was', 'were', 'be',
              'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but',
              'or', 'if', 'while', 'with', 'about', 'from', 'than', 'into'}

filtered_counts = Counter({word: count for word, count in word_counts.items()
                          if word not in stop_words})

print("\n========== WORD COUNT (AFTER REMOVING STOP WORDS) ==========")
print(f"{'Word':<20} {'Count':<10}")
print("-" * 30)
for word, count in sorted(filtered_counts.items()):
    print(f"{word:<20} {count:<10}")

# === Section 6: Total Statistics ===
print("\n========== STATISTICS ==========")
print(f"Total unique words: {len(word_counts)}")
print(f"Total words (including duplicates): {sum(word_counts.values())}")
print(f"Total unique words (no stop words): {len(filtered_counts)}")
print(f"Most frequent word: '{word_counts.most_common(1)[0][0]}' "
      f"({word_counts.most_common(1)[0][1]} times)")
