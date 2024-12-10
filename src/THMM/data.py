import os
import torch
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
from utils import BinarizerTHMM_TMM


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, mlb=None, expand_label=False):
        self.data = data
        self.texts = data['text']
        self.labels = data['labels']
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get text and corresponding labels
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text using BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Return a dictionary of tensors
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float),
        }
    
class TextDataset(CustomDataset):
    def __init__(self, df, tokenizer, max_len, mlb=None, expand_label=False):
        self.tokenizer = tokenizer
        self.data = df
        self.texts = list(df['text'])
        self.max_len = max_len
        self.expand_label = expand_label

        # Process labels
        self.labels_raw = self.data['label'].tolist()


        # Fit or use provided MultiLabelBinarizer
        if mlb is None:
            raise Exception("MultiLabelBinarizer is required")
        else:
            self.mlb = mlb

        # Binarize labels
        self.labels = self.mlb.transform(self.labels_raw)
        assert len(self.labels) == len(self.labels_raw)


class WOSdataset(CustomDataset):
    def __init__(self, df, tokenizer, max_len, mlb=None, expand_label=False):
        self.tokenizer = tokenizer
        self.data = df
        self.texts = list(df['raw'])
        self.max_len = max_len
        self.expand_label = expand_label

        # Process labels
        if expand_label:
            df['thmm_label'] = df['thmm_label'].apply(lambda x: [x[0], x])
        self.labels_raw = self.data['thmm_label'].tolist()


        # Fit or use provided MultiLabelBinarizer
        if mlb is None:
            raise Exception("MultiLabelBinarizer is required")
        else:
            self.mlb = mlb

        # Binarize labels
        self.labels = self.mlb.transform(self.labels_raw)
        assert len(self.labels) == len(self.labels_raw)


class USPTOdataset(CustomDataset):
    def __init__(self, df, tokenizer, max_len, mlb=None, expand_label=False):
        self.tokenizer = tokenizer
        self.data = df
        self.texts = list(df['text'])
        self.max_len = max_len
        self.expand_label = expand_label

        # Process labels
        self.labels_raw = self.data['Subclass_labels'].tolist()

        # Expand labels if required
        if self.expand_label:
            self.labels_raw = [
                list(set(sum([[j[0], j[:3], j] for j in tk], [])))
                for tk in self.labels_raw
            ]

        # Fit or use provided MultiLabelBinarizer
        if mlb is None:
            raise Exception("MultiLabelBinarizer is required")
        else:
            self.mlb = mlb

        # Binarize labels
        self.labels = self.mlb.transform(self.labels_raw)

        assert len(self.labels) == len(self.labels_raw)


class MIMICdataset(CustomDataset):
    def __init__(self, df, tokenizer, max_len, mlb=None, expand_label=False):
        self.tokenizer = tokenizer
        self.data = df
        self.texts = list(df['text'])
        self.max_len = max_len
        self.expand_label = expand_label

        # Process labels

        self.labels_raw = self.data['target'].tolist()
        print(self.labels_raw[0])

        # Expand labels if required
        if self.expand_label:
            self.labels_raw = [
                list(set(sum([[j.split('_')[0][:2], j.split('_')[0], j] for j in tk], [])))
                for tk in self.labels_raw
            ]

        # Fit or use provided MultiLabelBinarizer
        if mlb is None:
            raise Exception("MultiLabelBinarizer is required")
        else:
            self.mlb = mlb

        # Binarize labels
        self.labels = self.mlb.transform(self.labels_raw)

        assert len(self.labels) == len(self.labels_raw)


