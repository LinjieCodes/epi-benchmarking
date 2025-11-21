#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ROC & PR curves for 6 cell-line prediction results
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.interpolate import make_interp_spline


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

PLOT_CONFIG = {
    'dpi': 300,
    'figsize_roc': (6, 5),
    'figsize_pr': (6, 5),
}

COLORS = {
    'GM12878': '#1f77b4',
    'IMR90':   '#ff7f0e',
    'HeLa-S3': '#2ca02c',
    'HUVEC':   '#d62728',
    'K562':    '#9467bd',
    'NHEK':    '#8c564b'
}

def set_nature_style():
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'axes.grid': True,
        'grid.color': '.9',
        'grid.linewidth': 0.5,
        'font.size': 10,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': False,
        'legend.fontsize': 9,
        'figure.dpi': PLOT_CONFIG['dpi']
    })


def main():
    file_dict = {
        'GM12878': 'eModuleResults/GM12878_nonRedundant',
        'IMR90':   'eModuleResults/IMR90_nonRedundant',
        'HeLa-S3': 'eModuleResults/HeLa-S3_nonRedundant',
        'HUVEC':   'eModuleResults/HUVEC_nonRedundant',
        'K562':    'eModuleResults/K562_nonRedundant',
        'NHEK':    'eModuleResults/NHEK_nonRedundant',
    }


    roc_data, pr_data, f1_data = {}, {}, {}
    for cell, csv_path in file_dict.items():
        df = pd.read_csv(csv_path, header=None, sep=r'[,\t]', engine='python',skiprows=1)
        pred, label = df.iloc[:, 3].values, df.iloc[:, 4].values
        
        df = pd.DataFrame({'pred': pred, 'label': label})
        pos_cnt = df.label.sum()
        df = pd.concat([df[df.label==1],
                        df[df.label==0].sample(n=pos_cnt*20, random_state=42, replace=True)])
        pred, label = df['pred'].values, df['label'].values

        # ROC
        fpr, tpr, _ = roc_curve(label, pred)
        roc_auc = auc(fpr, tpr)
        roc_data[cell] = (fpr, tpr, roc_auc)
        
        # PR
        precision, recall, _ = precision_recall_curve(label, pred)
        aupr = average_precision_score(label, pred)
        pr_data[cell] = (recall, precision, aupr)
        
        prec = precision_score(label, pred)
        rec  = recall_score(label, pred)
        f1   = f1_score(label, pred)
        f1_data[cell] = (prec, rec, f1)
        

    set_nature_style()

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_roc'])
    for cell in roc_data:
        fpr, tpr, auc_score = roc_data[cell]
        ax.plot(fpr, tpr, color=COLORS[cell], lw=2.0,
                label=f'{cell} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], '#666666', lw=1.5, linestyle='--', alpha=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('eModule ROC Curves', fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('eModule_all_roc_curves.pdf', dpi=PLOT_CONFIG['dpi'],
                bbox_inches='tight', facecolor='white')
    plt.close()

    # ---- PR ----
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_pr'])
    for cell in pr_data:
        recall, precision, aupr_score = pr_data[cell]
        ax.plot(recall, precision, color=COLORS[cell], lw=2.0,
                label=f'{cell} (AUPR = {aupr_score:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('eModule PR Curves', fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=8)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout()
    plt.savefig('eModule_all_pr_curves.pdf', dpi=PLOT_CONFIG['dpi'],
                bbox_inches='tight', facecolor='white')
    plt.close()


    print('===== AUC / AUPR / Precision / Recall / F1 summary =====')
    for cell in roc_data:
        prec, rec, f1 = f1_data[cell]
        print(f"Cell line = {cell}  "
              f"Precision = {prec:.3f}  "
              f"Recall = {rec:.3f}  "
              f"F1 = {f1:.3f}")

if __name__ == '__main__':
    main()