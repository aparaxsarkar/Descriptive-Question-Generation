# pip install rouge

import pandas as pd
from rouge import Rouge
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

# Initialize the Rouge scorer
rouge_scorer = Rouge()

# Calculate ROUGE scores for each pair of entries
scores = []
for entry1_tokens, entry2_tokens in zip(tokenized_values_file1, tokenized_values_file2):
    # Using ROUGE-N metric, you can change 'N' to the desired value (e.g., ROUGE-1, ROUGE-2)
    rouge_scores = rouge_scorer.get_scores(" ".join(entry1_tokens), " ".join(entry2_tokens))
    scores.append(rouge_scores[0]['rouge-1']['f'])  # F1 score for ROUGE-1

# Display the ROUGE scores
df_scores = pd.DataFrame({'Ground Truth': values_file1, 'Generated Question': values_file2, 'ROUGE Score': scores})
print(df_scores)

mean_meteor_score = sum(scores) / len(scores)
print(f"Mean Rouge-1 Score: {mean_meteor_score}")