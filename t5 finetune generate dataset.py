# pip install --quiet -r requirements.txt

# pip install --quiet datasets
# pip install --quiet scikit-learn

from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def load_squad_dataset(dataset):
    df_dataset = pd.DataFrame(columns=['context', 'question', 'answer'])
    num_of_answer = 0
    for index, value in tqdm(enumerate(dataset)):
        context = value['context']
        question = value['question']
        answer = value['answers']['text'][0]
        number_of_words = len(answer.split())
        df_dataset.loc[num_of_answer] = [context] + [question] + [answer]
        num_of_answer = num_of_answer + 1
    return df_dataset

print('Downloading SQuAD dataset...')
train_dataset = load_dataset("squad", split='train')
valid_dataset = load_dataset("squad", split='validation')
print('train: ', len(train_dataset))
print('validation: ', len(valid_dataset))

pd.set_option('display.max_colwidth', None)
print('Loading df_train...')
df_train = load_squad_dataset(train_dataset)
print('Loading df_validation...')
df_validation = load_squad_dataset(valid_dataset)

print('Shuffling DataFrame...')
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_validation = df_validation.sample(frac=1, random_state=42).reset_index(drop=True)

print('Splitting into train, validation, and test sets...')
# 60-20-20
# df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
# df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# 70-15-15
# df_train, df_test = train_test_split(df_train, test_size=0.15, random_state=42)
# df_train, df_validation = train_test_split(df_train, test_size=(15/85), random_state=42)  # (15/85) x 0.85 = 0.15

# 80-10-10
df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=42)
df_train, df_validation = train_test_split(df_train, test_size=(1/9), random_state=42)  # (1/9) x 0.9 = 0.1

print('df_train.shape')
print(df_train.shape)
print('df_validation.shape')
print(df_validation.shape)
print('df_test.shape')
print(df_test.shape)

print('df_train.head():')
print(df_train.head())
print('df_validation.head():')
print(df_validation.head())
print('df_test.head():')
print(df_test.head())

print('Saving datasets as csv...')
dataset_save_path = 'datasets-80-10-10/'
if not os.path.exists(dataset_save_path):
    os.makedirs(dataset_save_path)

df_train.to_csv(dataset_save_path + 'squad_train.csv', index=False)
df_validation.to_csv(dataset_save_path + 'squad_validation.csv', index=False)
df_test.to_csv(dataset_save_path + 'squad_test.csv', index=False)