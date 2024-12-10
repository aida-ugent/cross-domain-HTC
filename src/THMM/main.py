import os
import torch
import random
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from collections import defaultdict
from networkx import DiGraph
from utils import create_hierarchical_tree, BinarizerTHMM_TMM
from models import TMM, THMM
from data import USPTOdataset, WOSdataset, TextDataset, MIMICdataset
from transformers import BertTokenizer
import pickle
import json

from tqdm import tqdm


# Fixing the random seeds
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configuration
config = {
  "model": "THMM",
  "output_dir": "./output/",
  "embedding_dir": "bert-base-uncased",
  "input_type": "CLS",
  "dense_layer_size": 256,
  "kernels": [3, 4, 5],
  "filter_size": 512,
  "epoch": 10,
  "batch_size": 2,
  "max_len": 256,
  "dropout_rate": 0.25,
  "learning_rate": 1e-05,
  "bert_trainable": True
}

MAX_LEN = config['max_len']
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = config['epoch']
LEARNING_RATE = config['learning_rate']


def get_data(data_dir, data='uspto'):
    tokenizer = BertTokenizer.from_pretrained(config["embedding_dir"])
    base_path = data_dir
    
    # Define dataset configurations
    dataset_configs = {
        'uspto': {
            'dataset_class': USPTOdataset,
            'expand_label': True,
            'max_len': MAX_LEN,
            'save_mapping': True
        },
        'uspto-10k': {
            'dataset_class': USPTOdataset,
            'expand_label': True,
            'max_len': MAX_LEN,
            'save_mapping': True
        },
        'uspto-100k': {
            'dataset_class': USPTOdataset,
            'expand_label': True,
            'max_len': MAX_LEN,
            'save_mapping': True
        },
        'wos': {
            'dataset_class': WOSdataset,
            'expand_label': True,
            'max_len': MAX_LEN
        },
        'nyt': {
            'dataset_class': TextDataset,  
            'expand_label': False,
            'max_len': MAX_LEN
        },
        'eurlexdc': {
            'dataset_class': TextDataset, 
            'expand_label': False,
            'max_len': MAX_LEN
        },
        'scihtc83': {
            'dataset_class': TextDataset,  
            'expand_label': False,
            'max_len': MAX_LEN
        },
        'scihtc800': {
            'dataset_class': TextDataset, 
            'expand_label': False,
            'max_len': MAX_LEN
        },
        'eurlex': {
            'dataset_class': TextDataset, 
            'expand_label': False,
            'max_len': MAX_LEN
        },
        'mimic3': {
            'dataset_class': MIMICdataset,
            'expand_label': True,
            'max_len': 512,
            'save_mapping': True
        }
    }
    
    if data not in dataset_configs:
        raise ValueError(f"Dataset {data} not supported")
    
    config = dataset_configs[data]
    
    # Load data and splits
    df_data = pd.read_feather(f"{base_path}/{data}/{data}.feather")
    df_splits = pd.read_feather(f"{base_path}/{data}/{data}_splits.feather")
    
    # Standardize label format (replace '.' with '_' in target labels)
    df_data['label'] = df_data['target'].apply(lambda x: [i.replace('.', '_') for i in x])
    
    # Split data based on splits file
    train_df = df_data[df_splits['split'] == 'train']
    val_df = df_data[df_splits['split'] == 'val']
    test_df = df_data[df_splits['split'] == 'test']
    
    # Load taxonomy and create graph
    all_labels, hiera = hiera_from_taxonomy(path=f"{base_path}/{data}/taxonomy.txt")
    graph = DiGraph(hiera)
    
    # Create binarizer
    binarizer = BinarizerTHMM_TMM()
    binarizer.fit([[x] for x in all_labels])
    
    # Save label mapping if specified
    if config.get('save_mapping', False):
        with open(f'./{data}-label2idx.pkl', 'wb') as f:
            pickle.dump(dict(zip(binarizer.mlb.classes_, range(len(binarizer.mlb.classes_)))), f)
    
    # Create datasets
    dataset_args = {
        'tokenizer': tokenizer,
        'mlb': binarizer,
        'max_len': config['max_len'],
        'expand_label': config['expand_label']
    }
    
    train_dataset = config['dataset_class'](df=train_df, **dataset_args)
    valid_dataset = config['dataset_class'](df=val_df, **dataset_args)
    test_dataset = config['dataset_class'](df=test_df, **dataset_args)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Dataset: {data}")
    print(f"Number of labels: {len(all_labels)}")
    print(f"Train/Val/Test splits: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    
    return train_loader, valid_loader, test_loader, binarizer, graph

def hiera_from_taxonomy(path):
    f = open(path, 'r')
    hiera = {}
    all_labels = []
    for line in f:
        split = line.strip().split('\t')
        split = [x.replace('.', '_') for x in split]
        all_labels.extend(split[1:])
        k = split[0]
        if split[0] == 'Root':
            k = 'ROOT'
        hiera[k] = split[1:]
    return all_labels,hiera

# Training function
def train_model(model, train_dataloader, val_dataloader, num_epochs=5, output_dir='./output'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = defaultdict(list)
    best_accuracy = 0
    best_loss = 100
    
    for epoch in range(num_epochs):
        model, train_acc, train_loss = train_one_epoch(model, train_dataloader, num_epochs, device, optimizer, criterion, epoch)
        # validation
        with torch.no_grad():
            val_acc, val_loss = evaluate_model(model, val_dataloader, device, criterion)
            if val_acc > best_accuracy: # save lowest loss -> worse on EurlexDC
                torch.save(model.state_dict(), output_dir + '/' + 'best_model_state.bin')
                best_accuracy = val_acc
                # best_loss = val_loss

        print(f'Epoch {epoch + 1}/{num_epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)


def evaluate(model, dataloader, output_dir='./output'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # get predictions   
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            predictions.append(outputs.cpu().numpy())
            targets.append(batch['labels'].cpu().numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    # save
    pickle.dump(predictions, open(output_dir + '/' + 'predictions.pkl', 'wb'))
    pickle.dump(targets, open(output_dir + '/' + 'targets.pkl', 'wb'))
    return predictions, targets


def evaluate_model(model, val_dataloader, device, criterion):
    losses = []
    total_loss = 0
    correct_predictions = 0
    num_samples = 0
    # set model to training mode (activate droput, batch norm)
    model.eval()
    # initialize the progress bar
    loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), 
                      leave=True)
    for batch_idx, batch in loop:
        labels = batch['labels'].to(device)
            # print(labels.shape)

        outputs = model(input_ids = batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device), token_type_ids = batch['token_type_ids'].to(device))
        
        preds = torch.argmax(outputs, dim=-1)
        labels = labels.argmax(dim=2)
        correct_predictions += torch.sum(preds == labels).cpu().numpy()
        num_samples += labels.numel()

        outputs_flat = outputs.view(-1, 2)
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)
        losses.append(loss.item())
        total_loss += loss.item()

    return float(correct_predictions)/num_samples, np.mean(losses)

def train_one_epoch(model, train_dataloader, num_epochs, device, optimizer, criterion, epoch):
    losses = []
    total_loss = 0
    correct_predictions = 0
    num_samples = 0

    model.train()
    # initialize the progress bar
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                      leave=True)
    for batch_idx, batch in loop:
        labels = batch['labels'].to(device)
            # print(labels.shape)

        optimizer.zero_grad()
        outputs = model(input_ids = batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device), token_type_ids = batch['token_type_ids'].to(device))
        
        preds = torch.argmax(outputs, dim=-1)
        labels = labels.argmax(dim=2)
        correct_predictions += torch.sum(preds == labels).cpu().numpy()
        num_samples += labels.numel()

        outputs_flat = outputs.view(-1, 2)
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return model, float(correct_predictions)/num_samples, np.mean(losses)


def main(args):
    train_loader, valid_loader, test_loader, binarizer, graph = get_data(args.data_dir, args.data)

    output_dir = config['output_dir'] + args.data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    thmm_model = THMM(config, binarizer, graph)

    print("\nTraining THMM model:")
    train_model(thmm_model, train_loader, valid_loader, num_epochs=EPOCHS, output_dir=output_dir)

    print("evaluating model")
    thmm_model.load_state_dict(torch.load(output_dir + '/' + 'best_model_state.bin'))
    thmm_model.eval()
    predictions, targets = evaluate(thmm_model, test_loader, output_dir=output_dir)
    print(predictions.shape, targets.shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='uspto')
    parser.add_argument('--data_dir', type=str, default='path/to/data')
    args = parser.parse_args()
    main(args)