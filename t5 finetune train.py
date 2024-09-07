import os
import time
import copy
import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm

# train_file_path = 'datasets-70-15-15/squad_train.csv'
# validation_file_path = 'datasets-70-15-15/squad_validation.csv'
# save_model_path = 'small/70-15-15 5e-5/BS8/model/'
# save_tokenizer_path = 'small/70-15-15 5e-5/BS8/tokenizer/'

# pretrained_model = 't5-small'
# # pretrained_model = 't5-base'

# args = argparse.Namespace()
# args.num_workers = 1
# args.batch_size = 8
# args.learning_rate = 5e-5
# args.eps = 1e-8
# args.weight_decay = 0.0


train_file_path = 'datasets-60-20-20/squad_train.csv'
validation_file_path = 'datasets-60-20-20/squad_validation.csv'
save_model_path = 'base/60-20-20 3e-5/BS16/model/'
save_tokenizer_path = 'base/60-20-20 3e-5/BS16/tokenizer/'

# pretrained_model = 't5-small'
pretrained_model = 't5-base'

args = argparse.Namespace()
args.num_workers = 1
args.batch_size = 16
args.learning_rate = 3e-5
args.eps = 1e-8
args.weight_decay = 0.0

class QGDataset(Dataset):

    def __init__(self, tokenizer, file_path, max_len_input=512, max_len_output=128):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(file_path)
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.context_column = 'context'
        self.answer_column = 'answer'
        self.question_column = 'question'
        self.inputs = []
        self.targets = []
        self._load_data()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids, 'target_mask': target_mask, 'labels': labels}

    def _load_data(self):
        for idx in tqdm(range(len(self.data))):

            context, answer, target = self.data.loc[idx, self.context_column], self.data.loc[idx, self.answer_column], self.data.loc[idx, self.question_column]
            # if len(str(answer).split()) >= 8:
            #     input_text = '<longanswer> %s <context> %s ' % (answer, context)
            # else:
            #     input_text = '<answer> %s <context> %s ' % (answer, context)
            input_text = '<answer> %s <context> %s ' % (answer, context)
            target = str(target)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.max_len_input,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_len_output,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


class T5FineTuner(pl.LightningModule):

    def __init__(self, model, tokenizer, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels, # decoder_input_ids included in lm_labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            # decoder_input_ids=batch['target_ids'],
            # decoder_attention_mask=batch['target_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        # logits = outputs.logits
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            # decoder_input_ids=batch['target_ids'],
            # decoder_attention_mask=batch['target_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        # logits = outputs.logits
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=args.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.args.batch_size, num_workers=args.num_workers, persistent_workers=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.args.learning_rate, eps=args.eps)


if __name__ == "__main__":
    
    start_time = time.time()
    pl.seed_everything(99)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print('Loading pre-trained model...')
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model).to(device)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model, model_max_length=512)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )

    print('Preparing dataset...')
    train_dataset = QGDataset(tokenizer, train_file_path)
    validation_dataset = QGDataset(tokenizer, validation_file_path)

    print('train_dataset: ', len(train_dataset))
    print('validation_dataset: ', len(validation_dataset))

    print ('Initializing model...')

    model = T5FineTuner(model, tokenizer, args)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',  # Use 'gpu' for GPU training
        # gpus=1,
        # gradient_clip_val=1.0,
        # auto_lr_find=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=1)]
        # progress_bar_refresh_rate=1,
        # print_nan_grads=True
    )
    # print('Run learning rate finder...')
    # lr_finder = trainer.tuner.lr_find(model, num_training=100)
    # args.learning_rate = lr_finder.suggestion()
    # print('Suggested lr: ', lr_finder.suggestion())

    print('Fine tuning...')
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model)
    # trainer.test()

    print('Saving model...')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_tokenizer_path):
        os.makedirs(save_tokenizer_path)
    model.model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_tokenizer_path)

    end_time = time.time() - start_time
    print('Total time: %s hours' % (end_time / 60 / 60))
    print('All done.')