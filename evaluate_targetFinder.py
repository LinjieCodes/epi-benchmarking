#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Six-cell-line TargetFinder evaluation  
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


PLOT_CONFIG = {
    'dpi': 300,
    'figsize_roc': (6, 5),
    'figsize_pr': (6, 5),
}


COLORS = {
    'GM12878': '#1f77b4',  # blue
    'IMR90':   '#ff7f0e',  # orange
    'HeLa-S3': '#2ca02c',  # green
    'HUVEC':   '#d62728',  # red
    'K562':    '#9467bd',  # purple
    'NHEK':    '#8c564b'   # brown
}


# ---------- plot style ----------
def set_nature_style():
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
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


# ---------- 10-fold CV for one cell line ----------
def cv_targetfinder(X, y, cell_name):
    """
    Returns  mean_fpr/tpr/recall/precision + AUC + AUPR
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    clf = GradientBoostingClassifier(
        n_estimators=4000, learning_rate=0.1, max_depth=5,
        max_features='log2', random_state=0
    )

    tprs, aucs = [], []
    precisions, auprs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    precisions_bin, recalls_bin, f1s_bin = [], [], []
    
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))

        # PR
        precision, recall, _ = precision_recall_curve(y_te, y_prob)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        auprs.append(average_precision_score(y_te, y_prob))
        
        y_pred = (y_prob >= 0.5).astype(int)
        precisions_bin.append(precision_score(y_te, y_pred))
        recalls_bin.append(recall_score(y_te, y_pred))
        f1s_bin.append(f1_score(y_te, y_pred))

    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_precision = np.mean(precisions, axis=0)
    return (mean_fpr, mean_tpr, np.mean(aucs), np.std(aucs),
            mean_recall, mean_precision, np.mean(auprs), np.std(auprs),
            np.mean(precisions_bin), np.std(precisions_bin),
            np.mean(recalls_bin), np.std(recalls_bin),
            np.mean(f1s_bin), np.std(f1s_bin))


# ---------- main pipeline ----------
def main():
    # 1. six cell lines and corresponding h5 paths
    cell_lines = ['GM12878', 'IMR90', 'HeLa-S3', 'HUVEC', 'K562', 'NHEK']
    data_dir = 'targetfinder-master/paper/targetfinder'   
    h5_dict = {c: f'{data_dir}/{c}/output-epw/training.h5' for c in cell_lines}

    # 2. non-predictor columns
    nonpred = ['enhancer_chrom', 'enhancer_start', 'enhancer_end',
               'promoter_chrom', 'promoter_start', 'promoter_end',
               'window_chrom', 'window_start', 'window_end', 'window_name',
               'active_promoters_in_window', 'interactions_in_window',
               'enhancer_distance_to_promoter', 'bin', 'label']

    # 3. evaluate each cell line
    roc_data, pr_data = {}, {}   
    results = []                 

    for cell in cell_lines:
        print(f'>>> Processing {cell} ...')
        df = pd.read_hdf(h5_dict[cell], 'training').set_index(['enhancer_name', 'promoter_name'])
        X = df.drop(columns=nonpred)
        y = df['label'].astype(int)

        (mean_fpr, mean_tpr, mean_auc, std_auc,
         mean_recall, mean_precision, mean_aupr, std_aupr,
         mean_pre, std_pre, mean_rec, std_rec, mean_f1, std_f1) = cv_targetfinder(X, y, cell)

        roc_data[cell] = (mean_fpr, mean_tpr, mean_auc)
        pr_data[cell]  = (mean_recall, mean_precision, mean_aupr)
        results.append({'Cell Line': cell,
                        'AUC':  f'{mean_auc:.3f}±{std_auc:.3f}',
                        'AUPR': f'{mean_aupr:.3f}±{std_aupr:.3f}',
                        'Precision': f'{mean_pre:.3f}±{std_pre:.3f}',
                        'Recall':    f'{mean_rec:.3f}±{std_rec:.3f}',
                        'F1':        f'{mean_f1:.3f}±{std_f1:.3f}'})
        #print(f'  AUC={mean_auc:.3f}±{std_auc:.3f}  AUPR={mean_aupr:.3f}±{std_aupr:.3f}')

    # 4. plot ROC and PR curves
    set_nature_style()

    # ---- ROC curves ----
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_roc'])
    for cell in cell_lines:
        fpr, tpr, auc_score = roc_data[cell]
        ax.plot(fpr, tpr, color=COLORS[cell], lw=2.0,
                label=f'{cell} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], '#666666', lw=1.5, linestyle='--', alpha=0.8)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('TargetFinder ROC Curves', fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('targetfinder_all_roc_curves.pdf', dpi=PLOT_CONFIG['dpi'],
                bbox_inches='tight', facecolor='white')
    plt.close()

    # ---- PR curves ----
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_pr'])
    for cell in cell_lines:
        recall, precision, aupr_score = pr_data[cell]
        ax.plot(recall, precision, color=COLORS[cell], lw=2.0,
                label=f'{cell} (AUPR = {aupr_score:.3f})')
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('TargetFinder PR Curves', fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig('targetfinder_all_pr_curves.pdf', dpi=PLOT_CONFIG['dpi'],
                bbox_inches='tight', facecolor='white')
    plt.close()

    # 5. print summary table
    print('\n===== TargetFinder 6-cell summary =====')
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == '__main__':
    main()