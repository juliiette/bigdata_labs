from pathlib import Path
from textblob import TextBlob

blob = TextBlob(Path("RomeoAndJuliet.txt").read_text())

print(blob.word_counts['juliet'])
print(blob.words.count('romeo\n'))




