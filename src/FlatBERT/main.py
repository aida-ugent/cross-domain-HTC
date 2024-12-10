# Standard library imports
import json
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from tqdm import tqdm

# Constants
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Hyperparameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-05
THRESHOLD = 0.5

class TextClassificationDataset(Dataset):
    """Dataset class for text classification tasks with transformer models."""
    
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int, text_col: str = 'text_short'):
        """
        Args:
            df: DataFrame containing the data
            tokenizer: Transformer tokenizer
            max_len: Maximum sequence length
            text_col: Name of the column containing the text data
        """
        self.tokenizer = tokenizer
        self.text = list(df[text_col])
        self.targets = df['onehot_labels'].tolist()
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = " ".join(str(self.text[index]).split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'text': text
        }

def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy loss with logits."""
    return torch.nn.BCEWithLogitsLoss()(logits, targets)

def get_data(data_dir='path/to/data', data='wos-141'):
    df = pd.read_feather(f'{data_dir}/{data}/{data}.feather', columns=['_id', 'text', 'target']).rename(columns={'text': 'text_short', 'target': 'label'})
    split_df = pd.read_feather(f'{data_dir}/{data}/{data}_splits.feather')
    df_train = df[split_df['split'] == 'train']
    df_valid = df[split_df['split'] == 'val']
    df_test = df[split_df['split'] == 'test']
    print(len(df_train), len(df_valid), len(df_test))
    all_labels = df_train['label'].tolist() + df_valid['label'].tolist() + df_test['label'].tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    df_train['onehot_labels'] = [list(x) for x in mlb.transform(df_train['label'].tolist())]
    df_valid['onehot_labels'] = [list(x) for x in mlb.transform(df_valid['label'].tolist())]
    df_test['onehot_labels'] = [list(x) for x in mlb.transform(df_test['label'].tolist())]
    label_mapping = {label: i for i, label in enumerate(mlb.classes_)}
    num_labels = len(label_mapping)
    label2id = label_mapping
    id2label = {v: k for k, v in label2id.items()}

    print(f"Number of labels: {num_labels}")
    return df_train, df_valid, df_test, num_labels, id2label, label2id
    

def train_model(training_loader: DataLoader, model: AutoModelForSequenceClassification, 
                optimizer: AdamW) -> Tuple[AutoModelForSequenceClassification, float, float]:
    """Train model for one epoch."""
    losses = []
    correct_predictions = 0
    num_samples = 0
    
    model.train()
    loop = tqdm(enumerate(training_loader), total=len(training_loader), leave=True)
    
    for _, data in loop:
        ids = data['input_ids'].to(model.device, dtype=torch.long)
        mask = data['attention_mask'].to(model.device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(model.device, dtype=torch.long)
        targets = data['targets'].to(model.device, dtype=torch.float)

        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs.logits, targets)
        losses.append(loss.item())

        outputs = torch.sigmoid(outputs.logits).cpu().detach().numpy().round()
        targets = targets.cpu().detach().numpy()
        correct_predictions += np.sum(outputs == targets)
        num_samples += targets.size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model, float(correct_predictions)/num_samples, np.mean(losses)

def eval_model(validation_loader: DataLoader, model: AutoModelForSequenceClassification, 
               optimizer: AdamW) -> Tuple[float, float]:
    losses = []
    correct_predictions = 0
    num_samples = 0
    
    model.eval()

    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(model.device, dtype=torch.long)
            mask = data['attention_mask'].to(model.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(model.device, dtype=torch.long)
            targets = data['targets'].to(model.device, dtype=torch.float)
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

            loss = loss_fn(outputs.logits, targets)
            losses.append(loss.item())

            outputs = torch.sigmoid(outputs.logits).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs == targets)
            num_samples += targets.size

    return float(correct_predictions)/num_samples, np.mean(losses)

def get_predictions(model: AutoModelForSequenceClassification, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Outputs:
      predictions - 
    """
    model = model.eval()
    
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
      for data in data_loader:
        ids = data["input_ids"].to(model.device, dtype = torch.long)
        mask = data["attention_mask"].to(model.device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(model.device, dtype = torch.long)
        targets = data["targets"].to(model.device, dtype = torch.float)
        
        outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        outputs = torch.sigmoid(outputs.logits).detach().cpu()
        preds = outputs.round()
        targets = targets.detach().cpu()

        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targets)
    
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)
    
    return predictions, prediction_probs, target_values

def train(train_data_loader: DataLoader, val_data_loader: DataLoader, num_labels: int, id2label: Dict[int, str], label2id: Dict[int, str], model_name: str = 'xlnet-base-cased', save_model_dir: str = './output/patentBERT'):
    # get model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=num_labels,
                                                            id2label=id2label,
                                                            label2id=label2id)
    print(model)
    model.to('cuda')

    # optimizer
    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)

    # Training loop
    history = defaultdict(list)
    best_accuracy = 0
    best_loss = 100
    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}/{EPOCHS}')
        model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)
        val_acc, val_loss = eval_model(val_data_loader, model, optimizer)

        print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        # save the best model
        if val_acc > best_accuracy or val_loss < best_loss:
            model.save_pretrained(save_model_dir)
            tokenizer.save_pretrained(save_model_dir+'_tokenizer')
            best_accuracy = val_acc
            best_loss = val_loss

def eval(test_data_loader: DataLoader, num_labels: int, id2label: Dict[int, str], label2id: Dict[int, str], model_dir: str = './output/patentBERT'):
    # evaluate the model on the test set
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=num_labels,
                                                            id2label=id2label,
                                                            label2id=label2id)
    model.to('cuda')   
    # get predictions
    predictions, prediction_probs, target_values = get_predictions(model, test_data_loader)
    # save predictions as numpy arrays
    predictions = predictions.numpy()
    prediction_probs = prediction_probs.numpy()
    target_values = target_values.numpy()
    # save predictions
    pickle.dump(predictions, open(model_dir + '/predictions.pkl', 'wb'))
    pickle.dump(prediction_probs, open(model_dir + '/prediction_probs.pkl', 'wb'))
    pickle.dump(target_values, open(model_dir + '/target_values.pkl', 'wb'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--model_dir", type=str, default='./output/bert')
    parser.add_argument("--data_dir", type=str, default='path/to/data')
    parser.add_argument("--data", type=str, default='uspto-2m')
    args = parser.parse_args()
    if args.data == 'mimiciii_clean':
        MAX_LEN = 512

    model_name = args.model_name
    model_dir = args.model_dir + '/' + args.data

    df_train, df_valid, df_test, num_labels, id2label, label2id = get_data(data_dir=args.data_dir, data=args.data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = TextClassificationDataset(df_train, tokenizer, MAX_LEN)
    valid_dataset = TextClassificationDataset(df_valid, tokenizer, MAX_LEN)
    test_dataset = TextClassificationDataset(df_test, tokenizer, MAX_LEN)
    print(f"train dataset: {len(train_dataset)}, valid dataset: {len(valid_dataset)}, test dataset: {len(test_dataset)}")

    # Data loaders
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_data_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    train(train_data_loader, val_data_loader, num_labels=num_labels, id2label=id2label, label2id=label2id, model_name=model_name, save_model_dir=model_dir)

    eval(test_data_loader, num_labels=num_labels, id2label=id2label, label2id=label2id, model_dir=model_dir)

