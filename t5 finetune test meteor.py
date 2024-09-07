# pip install nltk

import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize

# Load the CSV files
df1 = pd.read_csv('datasets-60-20-20/squad_test.csv')
df2 = pd.read_csv('base/60-20-20 3e-5/BS16/generated_questions.csv')

# Specify the column names you want to compare
column_name1 = 'question'
column_name2 = 'generated_questions'

# Get values from each column
values_file1 = df1[column_name1].dropna()
values_file2 = df2[column_name2].dropna()

# Tokenize both reference (df1) and hypothesis (df2) using word_tokenize
tokenized_values_file1 = [word_tokenize(entry) for entry in values_file1]
tokenized_values_file2 = [word_tokenize(entry) for entry in values_file2]

# Calculate METEOR scores for each pair of entries
scores = []
for entry1_tokens, entry2_tokens in zip(tokenized_values_file1, tokenized_values_file2):
    score = meteor_score.meteor_score([entry1_tokens], entry2_tokens)
    scores.append(score)

# Display the METEOR scores
df_scores = pd.DataFrame({'Ground Truth': values_file1, 'Generated Question': values_file2, 'METEOR Score': scores})
print(df_scores.head(10))

mean_meteor_score = sum(scores) / len(scores)
print(f"Mean Meteor Score: {mean_meteor_score}")