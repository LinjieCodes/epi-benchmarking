#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score
from tqdm import tqdm
from pathlib import Path

# Import project modules
from config import *
from data_loader import load_all_test_data
from optimized_data_preprocessing import create_optimized_dataset
# from model_dnabert2_embedding_small import EPIModel
from torch.utils.data import DataLoader
import torch.nn.functional as F



class MyDataset():
    
    def __init__(self, enhancers, promoters, labels):
        self.enhancers = enhancers
        self.promoters = promoters
        self.labels = labels
        
        self.enhancer_count = len(enhancers)
        self.promoter_count = len(promoters)
        self.label_count = len(labels)
        
    def __getitem__(self, idx):
        enhancer_sequence = str(self.enhancers[idx])
        promoter_sequence = str(self.promoters[idx])
        label = int(self.labels[idx])
        return enhancer_sequence, promoter_sequence, label
    
    def __len__(self):
        return self.label_count
    
    def count_lines_in_txt(self, file_path):
        with open(file_path, "r") as file:
            line_count = len(file.readlines())
        
        return line_count
    
def match_motif(pwm_matrix, sequence, threshold):
        motif_length = len(pwm_matrix)
        matches = []
        for i in range(len(sequence) - motif_length + 1):
            subsequence = sequence[i:i+motif_length]
            score = 1.0
            for j in range(motif_length):
                nucleotide = subsequence[j]
                if score < 0: break
                if nucleotide == 'A':
                    score *= pwm_matrix[j][0]
                elif nucleotide == 'C':
                    score *= pwm_matrix[j][1]
                elif nucleotide == 'G':
                    score *= pwm_matrix[j][2]
                elif nucleotide == 'T':
                    score *= pwm_matrix[j][3]
            if score >= threshold:  # Set a threshold to determine match or not
                matches.append(i)
        return matches

def replace_motif_with_N(sequence, matches, motif_length):
        modified_sequence = list(sequence)
        for match in matches:
            modified_sequence[match:match+motif_length] = 'N' * motif_length
        return ''.join(modified_sequence)

def simple_collate_fn(batch):

    # split batch into components
    enhancer_sequences = [item[0] for item in batch]
    promoter_sequences = [item[1] for item in batch]
    enhancer_features = [item[2] for item in batch]
    promoter_features = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    
    # pad sequences using pad_sequence
    padded_enhancer_sequences = torch.nn.utils.rnn.pad_sequence(enhancer_sequences, batch_first=True, padding_value=0)
    padded_promoter_sequences = torch.nn.utils.rnn.pad_sequence(promoter_sequences, batch_first=True, padding_value=0)
    
    # if padded length is still smaller than max length, pad it
    if padded_enhancer_sequences.size(1) < MAX_ENHANCER_LENGTH:
        padding_size = MAX_ENHANCER_LENGTH - padded_enhancer_sequences.size(1)
        padded_enhancer_sequences = F.pad(
            padded_enhancer_sequences, (0, padding_size), mode='constant', value=0
        )
    
    if padded_promoter_sequences.size(1) < MAX_PROMOTER_LENGTH:
        padding_size = MAX_PROMOTER_LENGTH - padded_promoter_sequences.size(1)
        padded_promoter_sequences = F.pad(
            padded_promoter_sequences, (0, padding_size), mode='constant', value=0
        )
    
    # directly stack feature tensors and labels
    padded_enhancer_features = torch.stack(enhancer_features)
    padded_promoter_features = torch.stack(promoter_features)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return padded_enhancer_sequences, padded_promoter_sequences, padded_enhancer_features, padded_promoter_features, labels


def evaluate_model(model, dataloader, device, cell_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {cell_name}"):
            enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            enhancer_features = enhancer_features.to(device)
            promoter_features = promoter_features.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # calculate metrics
    # unbalanced dataset, use 0.1 as threshold to calculate other metrics
    binary_preds = (all_preds >= 0.1).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    
    return {
        'cell_line': cell_name,
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'total_samples': len(all_labels),
        'positive_samples': np.sum(all_labels == 1),
        'negative_samples': np.sum(all_labels == 0)
    }


def main():
    # set the device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # load the pre-trained model
    model_path = os.path.join(PROJECT_ROOT, "save_model/train.pt")
    print(f"load model: {model_path}")
    
    # use weights_only=False to load model, because model contains custom classes
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    
    # load all test data
    test_data = load_all_test_data()
    
    # exclude ALL cell line
    cell_lines_to_evaluate = [cell for cell in TEST_CELL_LINES if cell != "ALL"]
    print(f"evaluate cell lines: {', '.join(cell_lines_to_evaluate)}")
    
    # store all results
    all_results = []
    
    # evaluate each cell line
    for cell_line in cell_lines_to_evaluate:
        if cell_line not in test_data:
            print(f"warning: test data for {cell_line} does not exist, skip")
            continue
            
        enhancers_test, promoters_test, labels_test = test_data[cell_line]
        
        # create dataset
        test_dataset = MyDataset(enhancers_test, promoters_test, labels_test)
        
        # extract raw sequences and labels
        enhancers_test_raw = [test_dataset[i][0] for i in range(len(test_dataset))]
        promoters_test_raw = [test_dataset[i][1] for i in range(len(test_dataset))]
        labels_test_raw = [test_dataset[i][2] for i in range(len(test_dataset))]
        
        # create optimized test dataset
        optimized_test_dataset = create_optimized_dataset(
            enhancers=enhancers_test_raw,
            promoters=promoters_test_raw,
            labels=labels_test_raw,
            cache_dir=os.path.join(CACHE_DIR, f"{cell_line}_test_cache"),
            use_cache=True
        )
        
        # create data loader
        test_loader = DataLoader(
            dataset=optimized_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )
        
        # evaluate model
        results = evaluate_model(model, test_loader, device, cell_line)
        all_results.append(results)
        
        # print results
        print(f"\n{cell_line} evaluation results:")
        print(f"  accuracy: {results['accuracy']:.4f}")
        print(f"  f1 score: {results['f1']:.4f}")
        print(f"  recall: {results['recall']:.4f}")
        print(f"  total samples: {results['total_samples']}")
        print(f"  positive samples: {results['positive_samples']}")
        print(f"  negative samples: {results['negative_samples']}")
    
    # convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.join(PROJECT_ROOT, 'evaluate'), exist_ok=True)
    
    # save results to CSV file
    output_path = os.path.join(PROJECT_ROOT, 'evaluate',"evaluate.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nevaluation results saved to: {output_path}")
    
    # calculate and print average metrics   
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    
    print("\naverage metrics:")
    print(f"  average accuracy: {avg_accuracy:.4f}")
    print(f"  average f1 score: {avg_f1:.4f}")
    print(f"  average recall: {avg_recall:.4f}")


if __name__ == "__main__":
    main()