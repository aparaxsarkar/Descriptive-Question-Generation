# pip install --quiet transformers
# pip install --quiet nltk.translate.bleu_score

import pandas as pd
import torch
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

class EvaluationDataset(Dataset):
        def __init__(self, tokenizer, file_path, max_len_input=512):
            self.tokenizer = tokenizer
            self.data = pd.read_csv(file_path)
            self.max_len_input = max_len_input
            self.context_column = 'context'
            self.answer_column = 'answer'
            self.question_column = 'question'
            self.inputs = []
            self._load_data()

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, index):
            source_ids = self.inputs[index]['input_ids'].squeeze()
            source_mask = self.inputs[index]['attention_mask'].squeeze()
            return {'source_ids': source_ids, 'source_mask': source_mask}

        def _load_data(self):
            for idx in tqdm(range(len(self.data))):
                context, answer = self.data.loc[idx, self.context_column], self.data.loc[idx, self.answer_column]

                input_text = '<answer> %s <context> %s ' % (answer, context)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input_text],
                    max_length=self.max_len_input,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                self.inputs.append(tokenized_inputs)

def main():
    save_model_path = 'base/60-20-20 3e-5/BS16/model/'
    save_tokenizer_path = 'base/60-20-20 3e-5/BS16/tokenizer/'

    args = argparse.Namespace()
    args.num_workers = 1
    args.batch_size = 8
    args.learning_rate = 3e-5
    args.eps = 1e-8
    args.weight_decay = 0.0

    model = T5ForConditionalGeneration.from_pretrained(save_model_path)
    tokenizer = T5Tokenizer.from_pretrained(save_tokenizer_path, model_max_length=512)

    test_file_path = 'datasets-60-20-20\squad_test.csv'
    df_eval = pd.read_csv(test_file_path)

    eval_dataset = EvaluationDataset(tokenizer, test_file_path)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    print('Using device:', device)

    model.to(device)

    predictions = []

    # model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)

            # Generate predictions
            generated_ids = model.generate(
                source_ids,
                attention_mask=source_mask,
                max_length=100,
            )

            # Decode the generated IDs to text
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            predictions.extend(generated_text)

    # Add the predictions to the evaluation dataframe
    df_eval['predicted_questions'] = predictions

    # Calculate BLEU score
    reference_list = df_eval['question'].apply(lambda x: [x.split()])
    candidate_list = df_eval['predicted_questions'].apply(lambda x: x.split())

    bleu_score = corpus_bleu(reference_list, candidate_list)
    print(f'BLEU Score: {bleu_score}')

    print(predictions[:9])

    df_generated = pd.DataFrame({'generated_questions': predictions})
    generated_csv = 'base/60-20-20 3e-5/BS16/generated_questions.csv'
    df_generated.to_csv(generated_csv, index=False)

if __name__ == "__main__":
    main()