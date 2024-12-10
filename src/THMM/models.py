
import os
import pandas as pd
import numpy as np
import shutil
import sys
import tqdm.notebook as tq
from collections import defaultdict

import torch
import torch.nn as nn

from transformers import BertModel


class ModelBase(nn.Module):
    def __init__(self, config, binarizer):
        super(ModelBase, self).__init__()
        self.bert = BertModel.from_pretrained(config['embedding_dir'])
        self.dropout_rate = config['dropout_rate']
        self.dense_layer_size = config['dense_layer_size']
        self.binarizer = binarizer
        
        for param in self.bert.parameters():
            param.requires_grad = config['bert_trainable']

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.pooler_output

class TMM(ModelBase):
    def __init__(self, config, binarizer):
        super(TMM, self).__init__(config, binarizer)
        self.num_classes = len(self.binarizer.mlb.classes_)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, self.dense_layer_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.dense_layer_size, self.dense_layer_size),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.dense_layer_size, 2)
            ) for _ in range(self.num_classes)
        ])

    def forward(self, input_ids, attention_mask, token_type_ids):
        doc_representation = super().forward(input_ids, attention_mask, token_type_ids)
        outputs = [classifier(doc_representation) for classifier in self.classifiers]
        return torch.stack(outputs, dim=1)
    

class ClassifierWithHiddenState(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(ClassifierWithHiddenState, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        hidden = self.dropout(self.relu(self.layer2(x)))
        output = self.layer3(hidden)
        return torch.cat([output, hidden], dim=1)


class THMM(ModelBase):
    def __init__(self, config, binarizer, graph):
        super(THMM, self).__init__(config, binarizer)
        self.graph = graph
        self.classifiers = nn.ModuleDict()
        self._create_classifiers()

    def _create_classifiers(self):
        for node in self.graph.nodes():
            if node != 'ROOT':
                if 'ROOT' in self.graph.predecessors(node):
                    input_size = 768
                    print(node)
                else:
                    input_size = 768 + self.dense_layer_size
                self.classifiers[node] = ClassifierWithHiddenState(
                    input_size=input_size,
                    hidden_size=self.dense_layer_size,
                    dropout_rate=self.dropout_rate
                )
            
    def forward(self, input_ids, attention_mask, token_type_ids):
        doc_representation = super().forward(input_ids, attention_mask, token_type_ids)
        outputs = {}
        
        def process_node(node, parent_hidden_state=None):
            if node == 'ROOT':
                for child in self.graph.successors(node):
                    process_node(child)
            else:
                if parent_hidden_state is not None:
                    input_tensor = torch.cat([doc_representation, parent_hidden_state], dim=1)
                else:
                    input_tensor = doc_representation
                
                output = self.classifiers[node](input_tensor)
                outputs[node] = output[:, :2]  # Extract output (first 2 elements)
                
                hidden_state = output[:, 2:]  # Extract hidden state (all but first 2 elements)
                
                for child in self.graph.successors(node):
                    process_node(child, hidden_state)
        
        process_node('ROOT')

        # Collect outputs in order of self.binarizer.mlb.classes_
        outputs_list = []
        for class_ in self.binarizer.mlb.classes_:
            outputs_list.append(outputs[class_])  # Each is of shape (batch_size, 2)

        # Stack outputs to get tensor of shape (batch_size, num_classes, 2)
        outputs_tensor = torch.stack(outputs_list, dim=1) 
        return outputs_tensor

