import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------- F1-score ----------
f1_lists = [
    [0.759, 0.891, 0.876, 0.814, 0.807, 0.893],   # eModule
    [0.647, 0.612, 0.766, 0.611, 0.732, 0.778],   # TargetFinder
    [0.790, 0.788, 0.846, 0.813, 0.847, 0.886]    # EPIPDLF
]

cell_lines = ['GM12878', 'IMR90',  'HeLa-S3', 'HUVEC', 'K562', 'NHEK']
models     = ['eModule', 'TargetFinder', 'EPIPDLF']

df = pd.DataFrame(f1_lists, index=models, columns=cell_lines)

plt.figure(figsize=(7, 3))
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap='Greens',              
        cbar_kws={'label': 'F1-score'},
        linewidths=.5,
        linecolor='lightgray',
        square=True,
        vmin=0.3, vmax=1.0         
)

ax.set_xlabel('')           
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.set_title('F1-score', fontweight='bold', pad=15)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('F1_heatmap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('Average F1-Score')
methods = ['eModule', 'TargetFinder', 'EPIPDLF']
for i in range(3):
    print(methods[i], round(np.mean(f1_lists[i]),3))
print()


# ---------- Pricision ----------
pricision_lists = [
    [0.844, 0.869, 0.862, 0.870, 0.858, 0.857],   # eModule
    [0.884, 0.854, 0.916, 0.885, 0.923, 0.929],   # TargetFinder
    [0.818, 0.817, 0.859, 0.836, 0.863, 0.894]    # EPIPDLF
]

cell_lines = ['GM12878', 'IMR90',  'HeLa-S3', 'HUVEC', 'K562', 'NHEK']
models     = ['eModule', 'TargetFinder', 'EPIPDLF']

df = pd.DataFrame(pricision_lists, index=models, columns=cell_lines)

plt.figure(figsize=(7, 3))
sns.set_style("whitegrid")          
plt.rcParams['font.size'] = 9

ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap='Greens',               
        cbar_kws={'label': 'Precision'},
        linewidths=.5,
        linecolor='lightgray',
        square=True,
        vmin=0.3, vmax=1.0         
)

ax.set_xlabel('')                  
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.set_title('Precision', fontweight='bold', pad=15)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('Precision_heatmap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('Average Pricision')
methods = ['eModule', 'TargetFinder', 'EPIPDLF']
for i in range(3):
    print(methods[i], round(np.mean(pricision_lists[i]),3))
print()


# ---------- Recall----------
recall_lists = [
    [0.689, 0.915, 0.890, 0.764, 0.762, 0.932],   # eModule
    [0.511, 0.478, 0.659, 0.467, 0.607, 0.669],   # TargetFinder
    [0.682, 0.681, 0.771, 0.712, 0.761, 0.824]    # EPIPDLF
]

cell_lines = ['GM12878', 'IMR90',  'HeLa-S3', 'HUVEC', 'K562', 'NHEK']
models     = ['eModule', 'TargetFinder', 'EPIPDLF']

df = pd.DataFrame(recall_lists, index=models, columns=cell_lines)

plt.figure(figsize=(7, 3))
sns.set_style("whitegrid")    
plt.rcParams['font.size'] = 9

ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap='Greens',        
        cbar_kws={'label': 'Recall'},
        linewidths=.5,
        linecolor='lightgray',
        square=True,
        vmin=0.3, vmax=1.0     
)

ax.set_xlabel('') 
ax.set_ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.set_title('Recall', fontweight='bold', pad=15)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('Recall_heatmap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('Average Recall')
methods = ['eModule', 'TargetFinder', 'EPIPDLF']
for i in range(3):
    print(methods[i], round(np.mean(recall_lists[i]),3))
print()